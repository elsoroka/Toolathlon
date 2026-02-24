import argparse
import asyncio
import json
from typing import Any, Dict, Optional

from utils.evaluation.evaluator import TaskEvaluator
from utils.status_manager import TaskStatusManager


def read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluation_status_from_pass_value(pass_value: Optional[bool]) -> Optional[str]:
    if pass_value is True:
        return "pass"
    if pass_value is False:
        return "fail"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Container-side evaluation for decoupled pipeline")
    parser.add_argument("--bundle_file", default="/workspace/dumps/task_bundle.json")
    parser.add_argument("--allow_resume", action="store_true")
    return parser.parse_args()


async def run_eval(bundle_file: str, allow_resume: bool = False) -> Dict[str, Any]:
    bundle = read_json_file(bundle_file)
    log_file = bundle["container_paths"]["log_file"]
    task_root = bundle["container_paths"]["task_root"]

    eval_res = await TaskEvaluator.evaluate_from_log_file(log_file, allow_resume=allow_resume)

    status_manager = TaskStatusManager(task_root)
    status_manager.update_evaluation(evaluation_status_from_pass_value(eval_res.get("pass")))

    return eval_res


def main() -> None:
    args = parse_args()
    eval_res = asyncio.run(run_eval(args.bundle_file, allow_resume=args.allow_resume))

    print("Evaluation finished.")
    print(f"Pass: {eval_res.get('pass')}")
    print(f"Details: {eval_res.get('details')}")
    if eval_res.get("failure") is not None:
        print(f"Failure: {eval_res.get('failure')}")

    raise SystemExit(0 if eval_res.get("pass") is True else 1)


if __name__ == "__main__":
    main()
