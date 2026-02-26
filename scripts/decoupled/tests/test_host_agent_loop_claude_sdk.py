import unittest

from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from scripts.decoupled.host_agent_loop_claude_sdk import (
    decide_task_status,
    parse_assistant_message,
    parse_user_message_content,
    resolve_claude_sdk_env,
)
from utils.roles.task_agent import TaskStatus


class HostAgentLoopClaudeSDKTests(unittest.TestCase):
    def test_resolve_claude_sdk_env_maps_auth_token(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "ANTHROPIC_BASE_URL": "https://api.example.com",
                "ANTHROPIC_AUTH_TOKEN": "token-a",
            }
        )
        self.assertEqual(resolved["ANTHROPIC_BASE_URL"], "https://api.example.com")
        self.assertEqual(resolved["ANTHROPIC_API_KEY"], "token-a")

    def test_resolve_claude_sdk_env_prefers_api_key(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "ANTHROPIC_API_KEY": "token-api",
                "ANTHROPIC_AUTH_TOKEN": "token-auth",
            }
        )
        self.assertEqual(resolved["ANTHROPIC_API_KEY"], "token-api")

    def test_parse_assistant_message_extracts_tool_calls(self) -> None:
        message = AssistantMessage(
            content=[
                TextBlock(text="hello"),
                ToolUseBlock(id="call_1", name="search", input={"q": "toolathlon"}),
            ],
            model="claude-sonnet-4-6",
        )

        entry, tool_calls, no_tool = parse_assistant_message(message)

        self.assertEqual(entry["role"], "assistant")
        self.assertEqual(entry["content"], "hello")
        self.assertEqual(entry["tool_calls_count"], 1)
        self.assertFalse(no_tool)
        self.assertEqual(tool_calls[0]["function"]["name"], "search")

    def test_parse_assistant_message_no_tool_call(self) -> None:
        message = AssistantMessage(
            content=[TextBlock(text="final answer")],
            model="claude-sonnet-4-6",
        )

        entry, tool_calls, no_tool = parse_assistant_message(message)

        self.assertEqual(entry["tool_calls_count"], 0)
        self.assertEqual(tool_calls, [])
        self.assertTrue(no_tool)

    def test_parse_user_message_content_blocks(self) -> None:
        text, blocks = parse_user_message_content(
            [
                ToolResultBlock(tool_use_id="call_1", content="done", is_error=False),
                TextBlock(text="ack"),
            ]
        )
        self.assertEqual(text, "done\nack")
        self.assertIsNotNone(blocks)
        assert blocks is not None
        self.assertEqual(blocks[0]["type"], "tool_result")

    def test_decide_task_status_prefers_no_tool_call(self) -> None:
        result_message = ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=2,
            session_id="sess_1",
            result="ok",
        )

        status, reason = decide_task_status(result_message, saw_stop_tool=True, saw_no_tool_call_turn=True)
        self.assertEqual(status, TaskStatus.SUCCESS)
        self.assertEqual(reason, "no_tool_call")

    def test_decide_task_status_max_turns(self) -> None:
        result_message = ResultMessage(
            subtype="error",
            duration_ms=100,
            duration_api_ms=80,
            is_error=True,
            num_turns=3,
            session_id="sess_2",
            result="Maximum turns reached",
        )

        status, reason = decide_task_status(result_message, saw_stop_tool=False, saw_no_tool_call_turn=False)
        self.assertEqual(status, TaskStatus.MAX_TURNS_REACHED)
        self.assertEqual(reason, "max_turns_reached")


if __name__ == "__main__":
    unittest.main()
