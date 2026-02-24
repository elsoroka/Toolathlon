import os
import tempfile
import unittest

import yaml

from scripts.decoupled.host_agent_loop import (
    build_gateway_runtime_mcp_config,
    expand_stop_tool_names,
    filter_local_tools,
)


class HostAgentLoopHelperTests(unittest.TestCase):
    def test_filter_local_tools_removes_ignored(self) -> None:
        tools = [
            "claim_done",
            "manage_context",
            "history",
            "handle_overlong_tool_outputs",
            "python_execute",
            "web_search",
        ]
        filtered = filter_local_tools(tools)
        self.assertEqual(filtered, ["python_execute", "web_search"])

    def test_expand_stop_tool_names_adds_gateway_prefix(self) -> None:
        expanded = expand_stop_tool_names(["local-claim_done", "local-custom"], "gw")
        self.assertIn("local-claim_done", expanded)
        self.assertIn("gw-local-claim_done", expanded)
        self.assertIn("local-custom", expanded)
        self.assertIn("gw-local-custom", expanded)

    def test_build_gateway_runtime_mcp_config_writes_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = build_gateway_runtime_mcp_config(
                runtime_dir=tmpdir,
                gateway_server_name="gateway_a",
                gateway_url="http://127.0.0.1:10086/sse",
            )
            config_path = os.path.join(runtime_dir, "gateway_sse.yaml")
            self.assertTrue(os.path.exists(config_path))

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["type"], "sse")
            self.assertEqual(config["name"], "gateway_a")
            self.assertEqual(config["params"]["url"], "http://127.0.0.1:10086/sse")


if __name__ == "__main__":
    unittest.main()
