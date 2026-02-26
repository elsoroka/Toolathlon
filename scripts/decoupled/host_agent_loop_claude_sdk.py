import argparse
import asyncio
import datetime
import json
import os
import traceback
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from claude_agent_sdk import query
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ContentBlock,
    PermissionResultAllow,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from scripts.decoupled.host_agent_loop import (
    build_host_task_config,
    expand_stop_tool_names,
    read_json_file,
)
from utils.general.helper import setup_proxy
from utils.roles.task_agent import TaskStatus
from utils.status_manager import TaskStatusManager
from utils.task_runner.runner import TaskRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host-side decoupled Claude SDK agent loop runner")
    parser.add_argument("--bundle_file", required=True)
    parser.add_argument("--gateway_url", required=True, help="SSE endpoint, e.g. http://127.0.0.1:10086/sse")
    parser.add_argument("--gateway_server_name", default="container_gateway")
    parser.add_argument("--model", default=None, help="Claude model id override")
    parser.add_argument("--max_turns", type=int, default=None)
    parser.add_argument("--with_proxy", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow_resume", action="store_true")
    return parser.parse_args()


def resolve_claude_sdk_env(source_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = source_env if source_env is not None else os.environ
    resolved: Dict[str, str] = {}

    base_url = env.get("ANTHROPIC_BASE_URL")
    if base_url:
        resolved["ANTHROPIC_BASE_URL"] = base_url

    api_key = env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
    if api_key:
        resolved["ANTHROPIC_API_KEY"] = api_key

    return resolved


def serialize_content_block(block: ContentBlock) -> Dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {"type": block.__class__.__name__}


def content_blocks_to_text(blocks: List[ContentBlock]) -> str:
    text_parts: List[str] = []
    for block in blocks:
        if isinstance(block, TextBlock) and block.text:
            text_parts.append(block.text)
        elif isinstance(block, ToolResultBlock) and isinstance(block.content, str) and block.content:
            text_parts.append(block.content)
    return "\n".join(text_parts).strip()


def build_tool_call_record(block: ToolUseBlock) -> Dict[str, Any]:
    return {
        "id": block.id,
        "type": "function",
        "function": {
            "name": block.name,
            "arguments": json.dumps(block.input, ensure_ascii=False),
        },
    }


def parse_assistant_message(message: AssistantMessage) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    tool_calls = [build_tool_call_record(block) for block in message.content if isinstance(block, ToolUseBlock)]
    content_blocks = [serialize_content_block(block) for block in message.content]
    text = content_blocks_to_text(message.content)

    entry: Dict[str, Any] = {
        "role": "assistant",
        "content": text,
        "content_blocks": content_blocks,
        "tool_calls_count": len(tool_calls),
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry, tool_calls, len(tool_calls) == 0


def parse_user_message_content(content: str | List[ContentBlock]) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    if isinstance(content, str):
        return content, None

    serialized = [serialize_content_block(block) for block in content]
    text = content_blocks_to_text(content)
    return text, serialized


def decide_task_status(
    result_message: Optional[ResultMessage],
    saw_stop_tool: bool,
    saw_no_tool_call_turn: bool,
) -> Tuple[TaskStatus, str]:
    if result_message is None:
        return TaskStatus.FAILED, "missing_result_message"

    if result_message.is_error:
        raw_result = (result_message.result or "").lower()
        if "max turn" in raw_result or "maximum turn" in raw_result:
            return TaskStatus.MAX_TURNS_REACHED, "max_turns_reached"
        return TaskStatus.FAILED, "sdk_result_error"

    # Decoupled hard requirement: no tool call is a termination condition.
    if saw_no_tool_call_turn:
        return TaskStatus.SUCCESS, "no_tool_call"
    if saw_stop_tool:
        return TaskStatus.SUCCESS, "stop_tool"

    return TaskStatus.FAILED, "termination_condition_not_met"


def build_agent_cost(result_message: Optional[ResultMessage]) -> Dict[str, Any]:
    if result_message is None:
        return {
            "total_cost": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
        }

    usage = result_message.usage or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "total_cost": round(float(result_message.total_cost_usd or 0.0), 6),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_requests": int(result_message.num_turns or 0),
    }


def build_key_stats(
    result_message: Optional[ResultMessage],
    assistant_turns: int,
    tool_calls_count: int,
) -> Dict[str, Any]:
    usage = result_message.usage or {} if result_message is not None else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)

    return {
        "interaction_turns": 1,
        "tool_calls": tool_calls_count,
        "agent_llm_requests": assistant_turns,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "sdk_num_turns": int(result_message.num_turns) if result_message is not None else 0,
    }


async def single_prompt_stream(prompt: str) -> AsyncIterator[Dict[str, Any]]:
    yield {
        "type": "user",
        "message": {
            "role": "user",
            "content": prompt,
        },
    }


async def allow_all_tools(_tool_name: str, _tool_input: Dict[str, Any], _ctx: Any) -> PermissionResultAllow:
    return PermissionResultAllow()


def write_traj_log(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


async def run_host_loop(args: argparse.Namespace) -> int:
    bundle = read_json_file(args.bundle_file)
    setup_proxy(args.with_proxy)

    eval_config_dict = bundle["eval_config"]
    _, agent_config, _ = TaskRunner.load_configs(eval_config_dict)
    task_config = build_host_task_config(bundle, agent_short_name=agent_config.model.short_name)

    task_config.stop.tool_names = expand_stop_tool_names(
        stop_tools=task_config.stop.tool_names or ["local-claim_done"],
        gateway_server_name=args.gateway_server_name,
    )
    stop_tool_name_set = set(task_config.stop.tool_names)

    max_turns = args.max_turns
    if max_turns is None:
        max_turns = int(bundle.get("max_steps_under_single_turn_mode") or task_config.max_steps_under_single_turn_mode or 100)

    model_name = args.model or agent_config.model.short_name
    sdk_env = resolve_claude_sdk_env()
    if "ANTHROPIC_API_KEY" not in sdk_env:
        raise RuntimeError("Missing ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) for Claude SDK host loop")

    status_manager = TaskStatusManager(task_config.task_root)
    status_manager.update_preprocess("done")
    status_manager.update_running("running")

    initial_run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completion_time = initial_run_time

    task_status = TaskStatus.FAILED
    termination_reason = "unknown"
    error_message: Optional[str] = None

    messages: List[Dict[str, Any]] = [{"role": "user", "content": task_config.task_str}]
    executed_tool_calls: List[Dict[str, Any]] = []
    assistant_turns = 0
    saw_stop_tool = False
    saw_no_tool_call_turn = False
    session_id: Optional[str] = None
    result_message: Optional[ResultMessage] = None

    try:
        os.makedirs(task_config.agent_workspace, exist_ok=True)

        options = ClaudeAgentOptions(
            model=model_name,
            system_prompt=task_config.system_prompts.agent,
            mcp_servers={
                args.gateway_server_name: {
                    "type": "sse",
                    "url": args.gateway_url,
                }
            },
            permission_mode="default",
            max_turns=max_turns,
            cwd=task_config.agent_workspace,
            env=sdk_env,
            can_use_tool=allow_all_tools,
        )

        async for message in query(prompt=single_prompt_stream(task_config.task_str), options=options):
            if isinstance(message, AssistantMessage):
                assistant_turns += 1
                entry, recent_tool_calls, no_tool_call_turn = parse_assistant_message(message)
                messages.append(entry)
                executed_tool_calls.extend(recent_tool_calls)
                saw_no_tool_call_turn = saw_no_tool_call_turn or no_tool_call_turn
                if any(call["function"]["name"] in stop_tool_name_set for call in recent_tool_calls):
                    saw_stop_tool = True

            elif isinstance(message, UserMessage):
                content_text, content_blocks = parse_user_message_content(message.content)
                entry: Dict[str, Any] = {"role": "user", "content": content_text}
                if content_blocks is not None:
                    entry["content_blocks"] = content_blocks
                if message.parent_tool_use_id is not None:
                    entry["parent_tool_use_id"] = message.parent_tool_use_id
                if message.tool_use_result is not None:
                    entry["tool_use_result"] = message.tool_use_result
                messages.append(entry)

            elif isinstance(message, SystemMessage):
                session_id = message.data.get("session_id") or session_id

            elif isinstance(message, ResultMessage):
                result_message = message
                session_id = message.session_id or session_id

            elif isinstance(message, StreamEvent):
                if args.debug and session_id is None:
                    session_id = message.session_id

        task_status, termination_reason = decide_task_status(
            result_message=result_message,
            saw_stop_tool=saw_stop_tool,
            saw_no_tool_call_turn=saw_no_tool_call_turn,
        )

        if task_status == TaskStatus.SUCCESS:
            status_manager.update_running("done")
        elif task_status == TaskStatus.MAX_TURNS_REACHED:
            status_manager.update_running("max_turn_exceeded")
        else:
            status_manager.update_running("fail")

    except Exception as e:
        if args.debug:
            traceback.print_exc()
        error_message = str(e)
        task_status = TaskStatus.FAILED
        termination_reason = "exception"
        status_manager.update_running("fail")
        print(f"Host Claude SDK loop failed: {e}")

    completion_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_payload: Dict[str, Any] = {
        "config": task_config.to_dict(),
        "request_id": str(uuid.uuid4()),
        "initial_run_time": initial_run_time,
        "completion_time": completion_time,
        "tool_calls": {
            "tools": executed_tool_calls,
            "tool_choice": agent_config.tool.tool_choice,
        },
        "status": task_status.value,
        "messages": messages,
        "key_stats": build_key_stats(
            result_message=result_message,
            assistant_turns=assistant_turns,
            tool_calls_count=len(executed_tool_calls),
        ),
        "agent_cost": build_agent_cost(result_message),
        "user_cost": {
            "total_cost": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
        },
        "resumed": bool(args.allow_resume),
        "session_id": session_id,
        "history_file": None,
        "termination_reason": termination_reason,
        "sdk_model": model_name,
        "stop_tool_detected": saw_stop_tool,
        "no_tool_call_detected": saw_no_tool_call_turn,
    }

    if result_message is not None:
        result_payload["sdk_result"] = {
            "subtype": result_message.subtype,
            "is_error": result_message.is_error,
            "num_turns": result_message.num_turns,
            "result": result_message.result,
            "usage": result_message.usage,
            "total_cost_usd": result_message.total_cost_usd,
        }

    if error_message is not None:
        result_payload["failure"] = error_message

    write_traj_log(task_config.log_file, result_payload)

    print(f"Host Claude SDK loop completed with status: {task_status.value} ({termination_reason})")
    return 0 if task_status == TaskStatus.SUCCESS else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run_host_loop(args)))


if __name__ == "__main__":
    main()
