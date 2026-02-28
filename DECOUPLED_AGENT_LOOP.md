# Decoupled Agent Loop

This mode separates the task environment from the agent framework.

## What Is Decoupled

In the original containerized run, preprocess, agent loop, and eval all happen inside one task container.

In the decoupled run:

1. The container does `preprocess`.
2. The container starts one MCP gateway and exposes the task tools over SSE.
3. The host runs the agent loop with a selectable agent framework.
4. The container does `eval`.

So the agent loop is no longer tied to the container runtime or sandbox. The container still owns the task environment and evaluation.

## Main Script

Use:

```bash
bash scripts/run_single_decoupled.sh \
  <task_dir> <runmode> <dump_path> <modelname> \
  [provider] [maxstep] [eval_config] [image_name] [agent_framework] [gateway_port]
```

Supported `agent_framework` values:

- `toolathlon_default`
- `claude_agent_sdk`

If `gateway_port` is omitted, the script auto-selects a free port.

## How It Works

The interface between container and host is intentionally narrow:

1. Container preprocess writes a task bundle.
2. Host agent reads the bundle for task prompt and runtime paths.
3. Host agent talks to the container only through the MCP gateway.
4. Host agent writes trajectory logs to the dump path.
5. Container eval reads the outputs and scores the task.

This is what makes it possible to swap the agent framework without changing the task container.

## Current Frameworks

### `toolathlon_default`

This is the existing Toolathlon scaffold built on the OpenAI Agents SDK.

It still uses the Toolathlon provider stack, so with `provider=unified` you should set:

```bash
export TOOLATHLON_OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export TOOLATHLON_OPENAI_API_KEY="your-key"
```

### `claude_agent_sdk`

This runs the host loop with Claude Agent SDK.

It does not use `TOOLATHLON_OPENAI_*`. It uses `ANTHROPIC_*` instead.

Example with OpenRouter:

```bash
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="YOUR_OPENROUTER_KEY"
export ANTHROPIC_API_KEY=""
export ANTHROPIC_DEFAULT_SONNET_MODEL="google/gemini-3-flash-preview"
```

Then run:

```bash
bash scripts/run_single_decoupled.sh \
  finalpool/find-alita-paper quickstart /tmp/dumps_claude_agent_sdk \
  sonnet unified 100 \
  scripts/formal_run_v0.json \
  lockon0927/toolathlon-task-image:1016beta \
  claude_agent_sdk
```

## Notes

- `scripts/formal_run_v0.json` is still the base run configuration template.
- `TOOLATHLON_MAX_TURNS_PER_TASK` can be used to override host-side max turns.
- The old one-container path is still available via `scripts/run_single_containerized.sh`.
