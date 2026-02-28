#!/bin/bash

# Formal entrypoint for decoupled single-task runs.
# Container responsibilities:
#   preprocess -> gateway -> eval
# Host responsibility:
#   agent loop selected by agent_framework
#
# Usage:
#   ./run_single_decoupled_formal.sh <task_dir> <runmode> <dump_path> <modelname> \
#     [provider] [maxstep] [eval_config] [image_name] [agent_framework] [gateway_port]
#
# Supported agent frameworks:
#   - toolathlon_default  -> existing OpenAI Agents SDK host loop
#   - claude_agent_sdk    -> Claude Agent SDK host loop
#
# Framework-specific controls are environment-variable driven.
# Current env vars:
#   - TOOLATHLON_AGENT_FRAMEWORK
#   - TOOLATHLON_MAX_TURNS_PER_TASK=<int>

set -e

task_dir_arg=$1
runmode=${2:-"normal"}
dump_path=${3:-"./dumps_quick_start"}
modelname=${4:-"anthropic/claude-sonnet-4.5"}
provider=${5:-"unified"}
maxstep=${6:-"100"}
eval_config=${7:-"scripts/formal_run_v0.json"}
image_name=${8:-"lockon0927/toolathlon-task-image:1016beta"}
agent_framework=${9:-"${TOOLATHLON_AGENT_FRAMEWORK:-toolathlon_default}"}
gateway_port=${10:-""}

if [ -z "$task_dir_arg" ] || [ -z "$runmode" ] || [ -z "$modelname" ]; then
    echo "Usage: $0 <task_dir> <runmode> <dump_path> <modelname> [provider] [maxstep] [eval_config] [image_name] [agent_framework] [gateway_port]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tool_call_mode="parallel"
host_loop_backend=""

case "$agent_framework" in
    toolathlon_default)
        host_loop_backend="openai"
        ;;
    claude_agent_sdk)
        host_loop_backend="claude_sdk"
        ;;
    *)
        echo "Unsupported agent_framework: $agent_framework"
        echo "Supported values: toolathlon_default, claude_agent_sdk"
        exit 1
        ;;
esac

echo "Agent framework: $agent_framework"
echo "Resolved host loop backend: $host_loop_backend"
if [ "$host_loop_backend" = "claude_sdk" ]; then
    echo "Claude tool call mode: parallel"
    echo "Claude permission mode: default"
    if [ -n "${TOOLATHLON_MAX_TURNS_PER_TASK:-}" ]; then
        echo "Task max turns override: ${TOOLATHLON_MAX_TURNS_PER_TASK}"
    fi
fi

exec bash "$SCRIPT_DIR/run_single_decoupled.sh" \
    "$task_dir_arg" \
    "$runmode" \
    "$dump_path" \
    "$modelname" \
    "$provider" \
    "$maxstep" \
    "$eval_config" \
    "$image_name" \
    "$gateway_port" \
    "$host_loop_backend" \
    "$tool_call_mode"
