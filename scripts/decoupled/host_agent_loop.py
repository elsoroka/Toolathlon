import argparse
import asyncio
import os
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import yaml

from utils.data_structures.task_config import SystemPrompts, TaskConfig
from utils.general.helper import (
    build_agent_model_provider,
    build_user_client,
    setup_proxy,
)
from utils.roles.task_agent import TaskAgent, TaskStatus
from utils.task_runner.runner import TaskRunner
from utils.task_runner.termination_checkers import default_termination_checker

from utils.openai_agents_monkey_patch.custom_run_impl import *
from utils.openai_agents_monkey_patch.custom_mcp_util import *


IGNORED_LOCAL_TOOLS = {"manage_context", "history", "handle_overlong_tool_outputs", "claim_done"}


def read_json_file(path: str) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_local_tools(tools: List[str]) -> List[str]:
    return [tool for tool in tools if tool not in IGNORED_LOCAL_TOOLS]


def expand_stop_tool_names(stop_tools: List[str], gateway_server_name: str) -> List[str]:
    expanded = set(stop_tools)
    for tool in stop_tools:
        expanded.add(f"{gateway_server_name}-{tool}")
    return sorted(expanded)


def decoupled_termination_checker(
    content: str,
    recent_tools: List[Dict],
    check_target: str = "user",
    user_stop_phrases: List[str] = [],
    agent_stop_tools: List[str] = [],
) -> bool:
    if default_termination_checker(
        content=content,
        recent_tools=recent_tools,
        check_target=check_target,
        user_stop_phrases=user_stop_phrases,
        agent_stop_tools=agent_stop_tools,
    ):
        return True

    # Decoupled hard requirement: stop when the agent makes no tool call in this round.
    if check_target == "agent" and len(recent_tools) == 0:
        return True

    return False


def build_gateway_runtime_mcp_config(
    runtime_dir: str,
    gateway_server_name: str,
    gateway_url: str,
) -> str:
    os.makedirs(runtime_dir, exist_ok=True)
    config_path = os.path.join(runtime_dir, "gateway_sse.yaml")
    config = {
        "type": "sse",
        "name": gateway_server_name,
        "params": {
            "url": gateway_url,
        },
        "cache_tools_list": False,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return runtime_dir


def build_host_task_config(bundle: Dict[str, Any], agent_short_name: str) -> TaskConfig:
    eval_config_dict = bundle["eval_config"]
    global_task_config = dict(eval_config_dict["global_task_config"])
    global_task_config["dump_path"] = bundle["host_paths"]["task_root"]
    global_task_config["direct_to_dumps"] = True

    task_config = TaskConfig.build(
        task_dir=bundle["task_dir"],
        agent_short_name=agent_short_name,
        global_task_config=global_task_config,
        single_turn_mode=bundle["single_turn_mode"],
        cn_mode=bundle["cn_mode"],
    )

    task_config.task_root = os.path.abspath(bundle["host_paths"]["task_root"])
    task_config.agent_workspace = os.path.abspath(bundle["host_paths"]["agent_workspace"])
    task_config.log_file = os.path.abspath(bundle["host_paths"]["log_file"])
    task_config.task_str = bundle["task_str"]
    task_config.launch_time = bundle["launch_time"]

    # Rebuild prompts with host workspace path and the original launch_time.
    task_config.system_prompts = SystemPrompts.build(task_config.task_dir, task_config.cn_mode)
    task_config.system_prompts.apply(
        task_config.agent_workspace,
        task_config.task_str,
        task_config.launch_time,
        task_config.single_turn_mode,
        task_config.cn_mode,
    )

    task_config.needed_local_tools = filter_local_tools(task_config.needed_local_tools or [])
    return task_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host-side decoupled agent loop runner")
    parser.add_argument("--bundle_file", required=True)
    parser.add_argument("--gateway_url", required=True, help="SSE endpoint, e.g. http://127.0.0.1:10086/sse")
    parser.add_argument("--gateway_server_name", default="container_gateway")
    parser.add_argument("--with_proxy", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow_resume", action="store_true")
    return parser.parse_args()


async def run_host_loop(args: argparse.Namespace) -> int:
    bundle = read_json_file(args.bundle_file)
    setup_proxy(args.with_proxy)

    eval_config_dict = bundle["eval_config"]
    mcp_config, agent_config, user_config = TaskRunner.load_configs(eval_config_dict)
    task_config = build_host_task_config(bundle, agent_short_name=agent_config.model.short_name)

    # Only keep one MCP server from container gateway on host side.
    runtime_dir = os.path.join(task_config.task_root, ".decoupled_runtime")
    mcp_config.server_config_path = build_gateway_runtime_mcp_config(
        runtime_dir=runtime_dir,
        gateway_server_name=args.gateway_server_name,
        gateway_url=args.gateway_url,
    )
    task_config.needed_mcp_servers = [args.gateway_server_name]

    # Stop on both local name and MCP-prefixed name.
    task_config.stop.tool_names = expand_stop_tool_names(
        stop_tools=task_config.stop.tool_names or ["local-claim_done"],
        gateway_server_name=args.gateway_server_name,
    )

    agent_model_provider = build_agent_model_provider(agent_config)
    user_client = build_user_client(user_config)

    task_agent = TaskAgent(
        task_config=task_config,
        agent_config=agent_config,
        agent_model_provider=agent_model_provider,
        user_config=user_config,
        user_client=user_client,
        mcp_config=mcp_config,
        termination_checker=partial(
            decoupled_termination_checker,
            user_stop_phrases=task_config.stop.user_phrases,
            agent_stop_tools=task_config.stop.tool_names,
        ),
        debug=args.debug,
        allow_resume=args.allow_resume,
        manual=False,
        single_turn_mode=task_config.single_turn_mode,
    )

    current_dir = os.path.abspath(os.getcwd())
    task_status = TaskStatus.FAILED

    try:
        # Preprocess is completed in container already.
        task_agent.status_manager.update_preprocess("done")

        await task_agent.setup_mcp_servers(local_token_key_session=bundle.get("local_token_key_session"))
        await task_agent.setup_agent()
        await task_agent.setup_user_simulator()

        os.chdir(task_config.agent_workspace)
        task_agent.status_manager.update_running("running")
        await task_agent.run_interaction_loop(abs_original_task_root=os.path.abspath(task_config.task_root))

        if task_agent.task_status not in [TaskStatus.MAX_TURNS_REACHED, TaskStatus.INTERRUPTED]:
            task_status = TaskStatus.SUCCESS
            task_agent.status_manager.update_running("done")
        else:
            task_status = task_agent.task_status
            if task_status == TaskStatus.MAX_TURNS_REACHED:
                task_agent.status_manager.update_running("max_turn_exceeded")

    except Exception as e:
        if args.debug:
            traceback.print_exc()
        task_status = TaskStatus.FAILED
        task_agent.status_manager.update_running("fail")
        print(f"Host loop failed: {e}")
    finally:
        os.chdir(current_dir)
        task_agent.task_status = task_status
        user_cost, agent_cost = task_agent.get_cost_summary()
        task_agent.user_cost = user_cost
        task_agent.agent_cost = agent_cost
        await task_agent.save_results()
        await task_agent.cleanup()

    print(f"Host loop completed with status: {task_status.value}")
    return 0 if task_status == TaskStatus.SUCCESS else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run_host_loop(args)))


if __name__ == "__main__":
    main()
