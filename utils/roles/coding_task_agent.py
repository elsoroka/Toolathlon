from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple, Callable, Awaitable
import os
import json
import uuid
import datetime
import traceback
from functools import partial
from enum import Enum
import pickle
from pathlib import Path

from agents import (
    Agent,
    RunConfig,
    Runner,
    Usage,
    ModelSettings,
    ToolCallItem,
    ModelProvider,
    ItemHelpers,
    RunContextWrapper,
    RunResult,
    RunResultStreaming,
)

from openai.types.responses import ResponseOutputMessage, ResponseFunctionToolCall, ResponseOutputText

from agents.exceptions import MaxTurnsExceeded
from agents.extensions.coding_planner_executor import (
    CodingPlannerOutput,
    CodingExecutorOutput,
    _DEFAULT_CODING_PLANNER_INSTRUCTIONS,
    _DEFAULT_CODING_EXECUTOR_INSTRUCTIONS,
)
from agents import function_tool as _function_tool
from agents.tool_context import ToolContext
from agents.items import ToolCallItem, MessageOutputItem

from utils.roles.context_managed_runner import ContextManagedRunner, _ServerConversationTracker
from utils.api_model.model_provider import ContextTooLongError

from utils.mcp.tool_servers import MCPServerManager
from utils.api_model.model_provider import calculate_cost, get_context_window
from utils.roles.user import User, UserRuntimeConfig
from utils.api_model.openai_client import AsyncOpenAIClientWithRetry
from utils.general.helper import copy_folder_contents, run_command, specifical_inialize_for_mcp
from utils.data_structures.task_config import TaskConfig
from utils.data_structures.agent_config import AgentConfig
from utils.data_structures.mcp_config import MCPConfig
from utils.data_structures.user_config import UserConfig
import shutil

import asyncio
import ast
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from utils.aux_tools.basic import tool_sleep, tool_done
from utils.aux_tools.ai_webpage_summary import tool_ai_webpage_summary
from utils.aux_tools.context_management_tools import context_management_tools
from utils.aux_tools.history_tools import history_tools
from utils.aux_tools.python_interpretor import tool_python_execute
from utils.aux_tools.web_search import tool_web_search
from utils.aux_tools.overlong_tool_manager import overlong_tool_tools

from utils.general.helper import print_color
from utils.status_manager import TaskStatusManager

from .task_agent import TaskAgent, local_tool_mappings, TaskStatus, TaskStatusManager, CustomJSONEncoder

# this is for exec()
@dataclass
class ExecutorObject:
    prompt: Callable
    call_tool: Callable
    return_result: Callable

class LlmPlannerError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class DummyResult:
    def __init__(self, items: list[MessageOutputItem]) -> None:
        self.new_items = items

class CodingTaskAgent(TaskAgent):
    # The CodingTaskAgent subclasses the TaskAgent and reimplements run() to support the coding pattern.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plan = None
        self._memory = "" # this is the executor memory


    # https://stackoverflow.com/questions/44859165/async-exec-in-python
    @classmethod
    async def aexec(cls, code:str, _globals:dict = None, _locals:dict = None):
        # Make an async function with the code and `exec` it
        #exec(
            #f'async def __ex(): ' +
            #''.join(f'\n {l}' for l in code.split('\n')), _globals, _globals)
#
        ## Get `__ex` from this context's local variables, call it and return the result
        #return await locals()['__ex']()
        code = compile(code, '<string>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        coroutine: Awaitable | None = eval(code, _globals, _locals)
        if coroutine is None:
            raise ValueError("The code did not return a coroutine")
        result = await coroutine
        print("result in aexec is", result)
        return result

    async def _build_tool_server_map(self):
        """Lazily build a map of tool_name -> (server, bare_tool_name). Called once and cached.

        Indexes tools under two keys so callers can use either form:
          - bare name:       "search_papers"
          - namespaced name: "arxiv_local-search_papers"
        """
        if hasattr(self, '_tool_server_map'):
            return self._tool_server_map
        self._tool_server_map = {}
        for server_name, server in self.mcp_manager.connected_servers.items():
            try:
                tools = await server.list_tools()
                for tool in tools:
                    bare = tool.name
                    namespaced = f"{server_name}-{bare}"
                    self._tool_server_map[bare] = (server, bare)
                    self._tool_server_map[namespaced] = (server, bare)
            except Exception as e:
                self._debug_print(f"Warning: could not list tools for server {server_name}: {e}")
        return self._tool_server_map

    # ------------- Tool calling ------------- #

    async def _call_tool(self, tool_name: str, **kwargs):
        """Call an MCP tool by name and return the text output, or None on error.

        Accepts both bare names ("search_papers") and namespaced names
        ("arxiv_local-search_papers") as used by the LLM planner.
        """
        tool_server_map = await self._build_tool_server_map()
        entry = tool_server_map.get(tool_name)
        if entry is None:
            self._debug_print(f"_call_tool: tool '{tool_name}' not found in any connected MCP server")
            return None

        server, bare_name = entry
        try:
            result = await server.call_tool(bare_name, arguments=kwargs)
        except Exception as e:
            self._debug_print(f"_call_tool: exception calling '{tool_name}': {e}")
            return None

        if result.isError:
            self._debug_print(f"_call_tool: tool '{tool_name}' returned error: {result.content}")
            return None

        if result.structuredContent is not None:
            return result.structuredContent

        if result.content:
            text = result.content[0].text
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text

        return None
    

    async def _call_executor(self, instructions:str, return_type: str = "str"):

        # Call the executor agent with the instructions, allowing it to call the mcp servers' tools and take multiple steps on its own.
        # get executor_result as a string
        self.agent.task = instructions
        executor_result = None
        try:
            result = await Runner.run(
                self.agent,
                instructions,
                run_config=RunConfig(model_provider=self.agent_model_provider),
                max_turns=self.agent_config.tool.max_inner_turns,
            )
            for raw_response in result.raw_responses:
                self.usage.add(raw_response.usage)
                self.stats["agent_llm_requests"] += 1
            executor_result = str(result.final_output) if result.final_output is not None else None
        except Exception as e:
            self._debug_print(f"Error calling executor on prompt: {instructions}. Exception: {e}")
            return None

        if not executor_result:
            return None

        # Try to cast the result to the right type
        try:
            if return_type == "str":
                return executor_result
            elif return_type == "int":
                return int(executor_result)
            elif return_type == "float":
                return float(executor_result)
            elif return_type == "bool":
                return bool(executor_result)
            elif return_type == "list":
                return [str(item).strip(" \"'") for item in str(executor_result).split(",")]
            else:
                raise ValueError(f"Unknown return type: {return_type}")
        except Exception as e:
            raise ValueError(f"Error converting executor result {executor_result} to type {return_type}. Exception: {e}")
    

    def _return_result(self, result: str):
        # Signal task completion — stores the result on self for execute_plan to retrieve.
        self._error = None
        self._debug_print(f"[executor.return_result] {result}")
        self.logs_to_record.append({
            "role": "assistant",
            "content": result,
            "tool_calls_count": 0,
        })
        message = ResponseOutputMessage(
            id="msg_2",
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=result,
                    type="output_text",
                    logprobs=[],
                )
            ],
        )
        dummy = DummyResult([MessageOutputItem(agent=self.agent, raw_item=message)])
        dummy.final_output = message
        self._executor_result = dummy


    async def execute_plan(self, abs_original_task_root: str) -> RunResult:
        # this is the parallel to task_agent.run_interaction_loop

        # Prepare the exec environment
        # Based on the tool list, we want to add functions with the names of the tool calls
        # but these have to be dynamically generated
        # We can use the tool_catalog to generate the functions
        exec_locals = {
            'executor' : ExecutorObject(
                prompt=self._call_executor,
                call_tool=self._call_tool,
                return_result=self._return_result,
            ),
            'result' : None,
        }
        # plan = f"""async def solve_task(executor):\n\t{"\n\t".join(lines)}\nawait solve_task(executor)"""

        lines = self._plan.splitlines()
        if not lines[-1].startswith("\t"):
            lines[-1] = "result = await solve_task(executor)"
            self._plan = "\n".join(lines)

        # This replaces the _run_interaction_loop
        # No try-catch because exceptions are caught in run()
        # print plan with line numbers
        self._debug_print(f"Executing plan: {"\n".join([f"{i+1}: {line}" for i, line in enumerate(self._plan.splitlines())])}")
        self._executor_result = None
        try:
            await self.aexec(self._plan, exec_locals, exec_locals)
            print("result is", self._executor_result)
            return self._executor_result
        
        except Exception as e:
            raise LlmPlannerError(f"Error when executing plan: {e}.")


    async def run(self) -> TaskStatus:
        """Run the whole task, including initialization, main loop, and saving results."""

        # Cache current working directory
        current_dir = os.path.abspath(os.getcwd())

        try:
            # Set log file and workspace dir
            self.task_config.log_file = os.path.join(self.task_config.task_root, "traj_log.json")
            self.task_config.agent_workspace = os.path.join(self.task_config.task_root, "workspace")

            # Preprocess status
            self.status_manager.update_preprocess("running")

            # Initialize workspace (skip if checkpoint will be used)
            if not await self.initialize_workspace():
                self.status_manager.update_preprocess("fail")
                return TaskStatus.FAILED

            self.status_manager.update_preprocess("done")
            
            # After preprocess, load task-specific local_token_key_session
            self.task_config.load_local_token_key_session()

            # Setup MCP servers
            await self.setup_mcp_servers(self.task_config.local_token_key_session)
            
            # Setup agent (LLM assistant)
            await self.setup_agent()

            tries = 0
            self._prev_error = None
            success = False
            while tries < 3 and not success:
                # Planner phase: generate a plan before the main loop (planner_executor pattern)
                await self._run_planner_phase(
                    _DEFAULT_PLANNER_INSTRUCTIONS=_DEFAULT_CODING_PLANNER_INSTRUCTIONS,
                    _DEFAULT_EXECUTOR_INSTRUCTIONS=_DEFAULT_CODING_EXECUTOR_INSTRUCTIONS)
                # the planner creates the _plan variable
                
                # Setup user simulator
                await self.setup_user_simulator()
                
                # Switch working dir to agent_workspace
                os.chdir(self.task_config.agent_workspace)
                self._debug_print(f"Switched working directory to {self.task_config.agent_workspace}")

                # Enter running status
                self.status_manager.update_running("running")

                # Main interaction loop
                try:
                    result = await self.execute_plan(os.path.abspath(self.task_config.task_root))
                    print("result is last", result)
                    if result.final_output is not None:
                        success = True
                    else:
                        raise ValueError(f"Plan didn't produce a final result: {result.final_output}")
                
                except LlmPlannerError as e:
                    self._prev_error = f"Error when executing plan: {e}. Traceback: {traceback.format_exc()}. Try again..."
                    tries += 1
                    if tries < 3:
                        self._debug_print(f"Error when executing plan: {self._prev_error}. Retrying...")
                        continue
                    else:
                        raise e

            # Switch back to the original cwd
            os.chdir(current_dir)
            self._debug_print(f"Switched back working directory to {current_dir}")
            
            # If not interrupted or max turns reached, mark done
            if self.task_status not in [TaskStatus.MAX_TURNS_REACHED, TaskStatus.INTERRUPTED]:
                self.task_status = TaskStatus.SUCCESS
                self.status_manager.update_running("done")
            elif self.task_status == TaskStatus.MAX_TURNS_REACHED:
                self.status_manager.update_running("max_turn_exceeded")
            
            # Remove checkpoint after successful completion
            if self.task_status == TaskStatus.SUCCESS:
                self._remove_checkpoint()
                
        except KeyboardInterrupt:
            self._debug_print("Task interrupted by user")
            if self.task_status != TaskStatus.INTERRUPTED:
                self.task_status = TaskStatus.INTERRUPTED
                
        except Exception as e:
            # max-turn logic updates the status in the interaction loop
            # but RuntimeError("Failed to get agent response...") brings us here,
            # so update status here as well
            self._debug_print("Error when running agent -", e)
            if self.debug:
                traceback.print_exc()
            if self.task_status == TaskStatus.MAX_TURNS_REACHED:
                self.status_manager.update_running("max_turn_exceeded")
            else:
                self.task_status = TaskStatus.FAILED
                self.status_manager.update_running("fail")
            
        finally:
            # Always restore working dir
            os.chdir(current_dir)
            self._debug_print(f"Switched back working directory to {current_dir}")

            # Gather final cost summary (updates token stats)
            user_cost, agent_cost = self.get_cost_summary()
            self.user_cost = user_cost
            self.agent_cost = agent_cost

            # Print cost/statistics summary (in English)
            self._debug_print(f"=== LLM-simulator ({self.user_config.model.short_name}) Cost Summary ===")
            for k, v in user_cost.items():
                self._debug_print(f"{k} : {v}")
            self._debug_print(f"=== Agent ({self.agent_config.model.short_name}) Cost Summary ===")
            for k, v in agent_cost.items():
                self._debug_print(f"{k} : {v}")
            self._debug_print("=== Key Statistics ===")
            for k, v in self.stats.items():
                self._debug_print(f"{k} : {v}")
            
            # Save final results to file
            await self.save_results()
            # Cleanup/close resources
            await self.cleanup()
            
        return self.task_status


    async def _run_planner_phase(self,
        _DEFAULT_PLANNER_INSTRUCTIONS=_DEFAULT_CODING_PLANNER_INSTRUCTIONS,
        _DEFAULT_EXECUTOR_INSTRUCTIONS=_DEFAULT_CODING_EXECUTOR_INSTRUCTIONS) -> None:
        """Run a single-turn planner agent to generate a Python plan.

        Builds a full tool catalog (name, description, parameter schemas) from the
        connected MCP servers, runs a planner Agent, then stores the resulting Python
        plan in self._plan. Does NOT rebuild self.agent — coding execution bypasses the
        LLM executor loop entirely.
        """
        self._debug_print("=== Running planner phase ===")

        # Build rich tool catalog directly from MCP servers so the planner sees
        # full parameter names and types, not just one-line descriptions.
        # Also warms up the tool->server map cache used by _call_tool.
        await self._build_tool_server_map()
        seen = set()
        lines = []
        for server_name, server in self.mcp_manager.connected_servers.items():
            try:
                tools = await server.list_tools()
            except Exception as e:
                self._debug_print(f"Warning: could not list tools for {server_name}: {e}")
                continue
            for tool in tools:
                namespaced = f"{server_name}-{tool.name}"
                if namespaced in seen:
                    continue
                seen.add(namespaced)
                desc = (tool.description or "").strip().splitlines()
                first_line = desc[0] if desc else ""
                lines.append(f"- {namespaced}: {first_line}" if first_line else f"- {namespaced}")
                # Append parameter info from inputSchema
                schema = tool.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                if props:
                    param_parts = []
                    for param_name, param_schema in props.items():
                        ptype = param_schema.get("type", "any")
                        req = " (required)" if param_name in required else ""
                        pdesc = param_schema.get("description", "")
                        pdesc_str = f" — {pdesc}" if pdesc else ""
                        param_parts.append(f"    - {param_name}: {ptype}{req}{pdesc_str}")
                    lines.extend(param_parts)
        tool_catalog = "\n".join(lines) if lines else "(no tools available)"

        # Create planner agent — no MCP servers, no tools, single turn
        planner_instructions = _DEFAULT_PLANNER_INSTRUCTIONS.format(
            tool_catalog=tool_catalog
        ) + """

## Execution environment notes
- The current working directory is already the task workspace directory. Do NOT prefix file paths with "workspace/". Use bare filenames or subdirectory-relative paths (e.g. "personal_info.md", not "workspace/personal_info.md").
- Do NOT chain methods directly on executor.prompt() or executor.call_tool() calls. Assign the result to a variable first, then call the method on the variable. For example:
  WRONG:  name = executor.prompt(...).strip()
  RIGHT:  name = executor.prompt(...)
          name = name.strip()

- Implement your plan in an async function called solve_task(executor), then call `await solve_task(executor)` in the last line.
"""
        if self._prev_error:
            planner_instructions += f"\n## Your last plan:\n{self._plan} resulted in the error:\n{self._prev_error}. Make sure to avoid this error in your plan."
        planner_agent: Agent = Agent(
            name="planner",
            instructions=planner_instructions,
            model=self.agent_model_provider.get_model(
                self.agent_config.model.real_name,
                debug=self.debug,
                short_model_name=self.agent_config.model.short_name,
            ),
            output_type=CodingPlannerOutput,
        )

        # Run planner (single turn, bare Runner — no history tracking needed)
        planner_result = await Runner.run(
            planner_agent,
            self.task_config.task_str,
            run_config=RunConfig(model_provider=self.agent_model_provider),
            max_turns=1,
        )

        # Account for planner token usage
        for raw_response in planner_result.raw_responses:
            self.usage.add(raw_response.usage)
            self.stats["agent_llm_requests"] += 1

        assert isinstance(planner_result.final_output, CodingPlannerOutput), (
            f"Planner returned unexpected output type: {type(planner_result.final_output)}"
        )
        self._plan = planner_result.final_output.plan
        self._memory = ""
        self._debug_print(f"=== Plan generated ===\n{self._plan}")

        # Callable so memory is re-evaluated fresh each turn
        def executor_instructions(ctx: RunContextWrapper, _agent: Agent) -> str:
            parts = [
                _DEFAULT_EXECUTOR_INSTRUCTIONS,
                f"## Tools\n{tool_catalog}",
                f"## Task\n{_agent.task}",
                f"## Memory of your previous actions\n{self._memory}",
            ]
                #f"## Additional Task Context\n{self.task_config.system_prompts.agent}",
            #]
            if self._memory:
                parts.append(f"## Memory of your previous actions\n{self._memory}")
            return "\n\n".join(parts)

        # Closure so writes go to self._memory (no shared context object needed)
        @_function_tool
        def update_memory(ctx: RunContextWrapper, note: str) -> str:
            """Append a note to your working memory (e.g. 'Completed step 2: found 3 files'). Call this to track progress on your task."""
            self._memory = (self._memory + "\n" + note).strip()
            return "Memory updated."

        self.agent = Agent(
            name="Assistant",
            instructions=executor_instructions,
            model=self.agent_model_provider.get_model(
                self.agent_config.model.real_name,
                debug=self.debug,
                short_model_name=self.agent_config.model.short_name,
            ),
            mcp_servers=[*self.mcp_manager.get_all_connected_servers()],
            tools=[*self.agent.tools, update_memory],
            hooks=self.agent_hooks,
            model_settings=ModelSettings(
                tool_choice=self.agent_config.tool.tool_choice,
                parallel_tool_calls=self.agent_config.tool.parallel_tool_calls,
                **{k: getattr(self.agent_config.generation, k)
                   for k in vars(self.agent_config.generation)},
            ),
        )
        self._debug_print("=== Executor agent rebuilt with prompt ===")
