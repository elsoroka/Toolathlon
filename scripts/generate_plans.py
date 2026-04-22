import asyncio
import json
import os
import sys
import tempfile
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure the Toolathlon root is on the path regardless of where the script is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

path = "/home/esoroka/Desktop/research/llm_stl_benchmark/toolathlon_benchmark/Toolathlon/tasks/finalpool/"

from agents.extensions.coding_planner_executor import (
    _DEFAULT_CODING_PLANNER_INSTRUCTIONS,
)
from utils.mcp.tool_servers import MCPServerManager

PLANNER_SUFFIX = """

## Execution environment notes
- The current working directory is already the task workspace directory. Do NOT prefix file paths with "workspace/". Use bare filenames or subdirectory-relative paths (e.g. "personal_info.md", not "workspace/personal_info.md").
- Do NOT chain methods directly on executor.prompt() or executor.call_tool() calls. Assign the result to a variable first, then call the method on the variable. For example:
  WRONG:  name = executor.prompt(...).strip()
  RIGHT:  name = executor.prompt(...)
          name = name.strip()

- Implement your plan in an async function called solve_task(executor), then call `await solve_task(executor)` in the last line.
"""

async def build_tool_catalog(mcp_manager: MCPServerManager) -> str:
    seen = set()
    lines = []
    for server_name, server in mcp_manager.connected_servers.items():
        try:
            tools = await server.list_tools()
        except Exception as e:
            print(f"Warning: could not list tools for {server_name}: {e}")
            continue
        for tool in tools:
            namespaced = f"{server_name}-{tool.name}"
            if namespaced in seen:
                continue
            seen.add(namespaced)
            desc = (tool.description or "").strip().splitlines()
            first_line = desc[0] if desc else ""
            lines.append(f"- {namespaced}: {first_line}" if first_line else f"- {namespaced}")
            schema = tool.inputSchema or {}
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            for param_name, param_schema in props.items():
                ptype = param_schema.get("type", "any")
                req = " (required)" if param_name in required else ""
                pdesc = param_schema.get("description", "")
                pdesc_str = f" — {pdesc}" if pdesc else ""
                lines.append(f"    - {param_name}: {ptype}{req}{pdesc_str}")
    return "\n".join(lines) if lines else "(no tools available)"


async def process_task(task_id: str, task_dir: str) -> dict | None:
    task_md = os.path.join(task_dir, "docs", "task.md")
    task_config_path = os.path.join(task_dir, "task_config.json")

    if not os.path.exists(task_md) or not os.path.exists(task_config_path):
        return None

    with open(task_md) as f:
        content = f.read()
    with open(task_config_path) as f:
        task_config = json.load(f)

    needed_servers = task_config.get("needed_mcp_servers", [])

    with tempfile.TemporaryDirectory() as tmpdir:
        mcp_manager = MCPServerManager(
            agent_workspace=tmpdir,
            config_dir="configs/mcp_servers",
        )
        try:
            await mcp_manager.connect_servers(needed_servers)
        except Exception as e:
            print(f"Warning for {task_id}: {e} — continuing with partial servers")
        tool_catalog = await build_tool_catalog(mcp_manager)
        await mcp_manager.disconnect_servers()

    planner_instructions = _DEFAULT_CODING_PLANNER_INSTRUCTIONS.format(
        tool_catalog=tool_catalog
    ) + PLANNER_SUFFIX

    return {
        "task_id": task_id,
        "prompt": content,
        "tool_catalog": tool_catalog,
        "planner_instructions": planner_instructions,
    }


async def main():
    prompts = []
    for task_id in tqdm(os.listdir(path)):
        task_dir = os.path.join(path, task_id)
        if not os.path.isdir(task_dir):
            print(f"Skipping {task_id}: not a directory")
            continue
        result = await process_task(task_id, task_dir)
        if result:
            prompts.append(result)
        else:
            print(f"Skipping {task_id}: missing task.md or task_config.json")
        
    with open("toolathlon_prompts.jsonl", "w") as f:
        for entry in tqdm(prompts):
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(prompts)} prompts to toolathlon_prompts.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
