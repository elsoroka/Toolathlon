import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.decoupled.container_eval import (
    evaluation_status_from_pass_value,
    remap_dump_line_paths_to_container,
    run_eval,
)


class ContainerEvalTests(unittest.TestCase):
    def test_evaluation_status_mapping(self) -> None:
        self.assertEqual(evaluation_status_from_pass_value(True), "pass")
        self.assertEqual(evaluation_status_from_pass_value(False), "fail")
        self.assertIsNone(evaluation_status_from_pass_value(None))

    def test_remap_dump_line_paths_to_container(self) -> None:
        bundle = {
            "container_paths": {
                "task_root": "/workspace/dumps/task",
                "agent_workspace": "/workspace/dumps/task/workspace",
                "log_file": "/workspace/dumps/task/traj_log.json",
            }
        }
        dump_line = {
            "config": {
                "task_root": "/root/host/task",
                "agent_workspace": "/root/host/task/workspace",
                "log_file": "/root/host/task/traj_log.json",
            }
        }
        mapped = remap_dump_line_paths_to_container(dump_line, bundle)
        self.assertEqual(mapped["config"]["task_root"], "/workspace/dumps/task")
        self.assertEqual(
            mapped["config"]["agent_workspace"], "/workspace/dumps/task/workspace"
        )
        self.assertEqual(
            mapped["config"]["log_file"], "/workspace/dumps/task/traj_log.json"
        )


class RunEvalTests(unittest.IsolatedAsyncioTestCase):
    @patch("scripts.decoupled.container_eval.TaskStatusManager")
    @patch("scripts.decoupled.container_eval.write_json_file")
    @patch("scripts.decoupled.container_eval.TaskEvaluator.evaluate_one", new_callable=AsyncMock)
    @patch("scripts.decoupled.container_eval.os.path.exists")
    @patch("scripts.decoupled.container_eval.read_json_file")
    async def test_run_eval_updates_status_manager(
        self,
        mock_read_json: MagicMock,
        mock_exists: MagicMock,
        mock_evaluate_one: AsyncMock,
        mock_write_json: MagicMock,
        mock_status_manager_cls: MagicMock,
    ) -> None:
        mock_exists.return_value = False
        mock_read_json.side_effect = [
            {
                "container_paths": {
                    "log_file": "/workspace/dumps/traj_log.json",
                    "task_root": "/workspace/dumps",
                    "agent_workspace": "/workspace/dumps/workspace",
                }
            },
            {
                "config": {
                    "task_root": "/root/host",
                    "agent_workspace": "/root/host/workspace",
                    "log_file": "/root/host/traj_log.json",
                }
            },
        ]
        mock_evaluate_one.return_value = {"pass": False, "details": "not pass"}
        mock_status_manager = MagicMock()
        mock_status_manager_cls.return_value = mock_status_manager

        result = await run_eval("/workspace/dumps/task_bundle.json", allow_resume=True)

        self.assertEqual(mock_read_json.call_count, 2)
        called_dump_line = mock_evaluate_one.await_args.args[0]
        self.assertEqual(called_dump_line["config"]["task_root"], "/workspace/dumps")
        self.assertEqual(
            called_dump_line["config"]["agent_workspace"], "/workspace/dumps/workspace"
        )
        mock_write_json.assert_called_once()
        mock_status_manager_cls.assert_called_once_with("/workspace/dumps")
        mock_status_manager.update_evaluation.assert_called_once_with("fail")
        self.assertEqual(result["pass"], False)


if __name__ == "__main__":
    unittest.main()
