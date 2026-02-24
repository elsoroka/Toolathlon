import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.decoupled.container_eval import (
    evaluation_status_from_pass_value,
    run_eval,
)


class ContainerEvalTests(unittest.TestCase):
    def test_evaluation_status_mapping(self) -> None:
        self.assertEqual(evaluation_status_from_pass_value(True), "pass")
        self.assertEqual(evaluation_status_from_pass_value(False), "fail")
        self.assertIsNone(evaluation_status_from_pass_value(None))


class RunEvalTests(unittest.IsolatedAsyncioTestCase):
    @patch("scripts.decoupled.container_eval.TaskStatusManager")
    @patch("scripts.decoupled.container_eval.TaskEvaluator.evaluate_from_log_file", new_callable=AsyncMock)
    @patch("scripts.decoupled.container_eval.read_json_file")
    async def test_run_eval_updates_status_manager(
        self,
        mock_read_json: MagicMock,
        mock_evaluate: AsyncMock,
        mock_status_manager_cls: MagicMock,
    ) -> None:
        mock_read_json.return_value = {
            "container_paths": {
                "log_file": "/workspace/dumps/traj_log.json",
                "task_root": "/workspace/dumps",
            }
        }
        mock_evaluate.return_value = {"pass": False, "details": "not pass"}
        mock_status_manager = MagicMock()
        mock_status_manager_cls.return_value = mock_status_manager

        result = await run_eval("/workspace/dumps/task_bundle.json", allow_resume=True)

        mock_evaluate.assert_awaited_once_with(
            "/workspace/dumps/traj_log.json", allow_resume=True
        )
        mock_status_manager_cls.assert_called_once_with("/workspace/dumps")
        mock_status_manager.update_evaluation.assert_called_once_with("fail")
        self.assertEqual(result["pass"], False)


if __name__ == "__main__":
    unittest.main()
