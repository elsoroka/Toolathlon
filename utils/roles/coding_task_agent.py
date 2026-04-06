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
)

from openai.types.responses import ResponseOutputMessage, ResponseFunctionToolCall

from agents.exceptions import MaxTurnsExceeded
from agents.extensions.coding_planner_executor import (
    CodingPlannerOutput,
    CodingExecutorOutput,
    _DEFAULT_CODING_PLANNER_INSTRUCTIONS,
    _DEFAULT_CODING_EXECUTOR_INSTRUCTIONS,
)
from agents import function_tool as _function_tool
from agents import RunResult
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
        await coroutine

    def _call_tool(self, tool_name:str, **kwargs):
        # return a tool call as if it was made by an LLM and return a RunResult with no usage
        # and a tool call in new_items
        raw_item = ResponseFunctionToolCall(
            id=str(uuid.uuid4()),
            call_id=str(uuid.uuid4()),
            type="function_call",
            name=tool_name,
            arguments=str(kwargs),
        )
        new_items = [ToolCallItem(agent=self.agent, raw_item=raw_item)]
        # in here we need to run the tool call!!
        # final_output = ???
        
        return RunResult(
            input=f"{tool_name}({kwargs})",
            new_items=new_items,
            raw_responses=[],
            final_output=final_output,
            _last_agent=self.agent,
            input_guardrail_results=[],
            output_guardrail_results=[],
            tool_input_guardrail_results=[],
            tool_output_guardrail_results=[],
            context_wrapper=None,
        )
    
    async def _call_executor(self, prompt:str, return_type: str = "str"):
        try:
            executor_result = await Runner.run(
                self.agent,
                prompt,
                run_config=RunConfig(model_provider=self.agent_model_provider),
                max_turns=self.agent_config.generation.max_turns,
            )
        except Exception as e:
            self._debug_print(f"Error calling executor on prompt: {prompt}. Exception: {e}")
            return None

        # Try to cast the result to the right type
        try:
            if return_type == "str":
                return executor_result.final_output.text
            elif return_type == "int":
                return int(executor_result.final_output.text)
            elif return_type == "float":
                return float(executor_result.final_output.text)
            elif return_type == "bool":
                return bool(executor_result.final_output.text)
            elif return_type == "list":
                return return_type(executor_result.final_output.text)
            else:
                self._debug_print(f"Unknown return type: {return_type}")
                return None
        except Exception as e:
            self._debug_print(f"Error converting executor result {executor_result.final_output.text} to type {return_type}. Exception: {e}")
            return None

    def _return_result(self, result:str):
        # make a fake LLM chat response with the result
        return RunResult(
            input=self.plan,
            new_items=[], 
            raw_responses=[],
            final_output=ResponseOutputMessage(role="assistant", content=result),
            _last_agent=self.agent,
            input_guardrail_results=[], 
            output_guardrail_results=[], 
            tool_input_guardrail_results=[], 
            tool_output_guardrail_results=[],
            context_wrapper=None,
        )

    async def execute_plan(self, abs_original_task_root: str) -> None:

        # Prepare the exec environment
        # Based on the tool list, we want to add functions with the names of the tool calls
        # but these have to be dynamically generated
        # We can use the tool_catalog to generate the functions
        exec_locals = {'executor' : ExecutorObject(prompt=self._call_executor, call_tool=self._call_tool, return_result=self._return_result)}
        
        # handle this dumb thing
        plan = self._plan.replace("executor.prompt", "await executor.prompt")
        # is there a function?
        if plan.startswith("def"):
            # get the function name
            function_name = plan.split("def")[1].split("(")[0].strip()
            plan.replace("def", "async def")
            if not function_name in plan.splitlines()[-1]:
                # append running the function to the plan
                plan += f"await {function_name}()"
            else:
                lines = plan.splitlines()
                lines[-1].replace(function_name, f"await {function_name}")
                plan = "\n".join(lines)
        
        # there isn't a function, so wrap the plan in one
        lines = plan.splitlines()
        plan = f"""async def solve_task(executor):\n\t{"\n\t".join(lines)}\nawait solve_task(executor)"""
        
        # This replaces the _run_interaction_loop
        # No try-catch because exceptions are caught in run()
        self._debug_print(f"Executing plan: {plan}")
        await self.aexec(plan, exec_locals, exec_locals)


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
            await self.execute_plan(os.path.abspath(self.task_config.task_root))

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