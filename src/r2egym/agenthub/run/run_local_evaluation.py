import argparse
import json
import pathlib

from r2egym.agenthub.runtime.local import LocalRuntime
from r2egym.commit_models.diff_classes import ParsedCommit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness for the given dataset and prediction.")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to a JSON file with the dataset",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--instance_id",
        type=str,
        help="Instance ID to run",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--predictions_path",
        type=str,
        help="Path to predictions file in SWE-bench format - if 'gold', uses gold predictions",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Folder to store output files",
    )

    args = parser.parse_args()

    with open(args.dataset, "r") as fin:
        for line in fin:
            row = json.loads(line)
            if row["instance_id"] == args.instance_id:
                dataset_row = row
                break
        else:
            raise ValueError(f"Could not find instance_id {args.instance_id} in dataset {args.dataset}")

    if args.predictions_path == "gold":
        commit = ParsedCommit(**json.loads(dataset_row["parsed_commit_content"]))
        patch = commit.get_patch()
    else:
        with open(args.predictions_path, "r") as fin:
            for line in fin:
                row = json.loads(line)
                if row["instance_id"] == args.instance_id:
                    patch = row["model_patch"]
                    break
            else:
                raise ValueError(
                    f"Could not find instance_id {args.instance_id} in predictions file {args.predictions_path}"
                )

    test_output = None

    if not patch:
        print("Empty patch.")
        report = {"resolved": False, "patch_exists": False, "patch_successfully_applied": False}
    else:
        runtime = LocalRuntime(dataset_row)

        print("Applying patch...")
        apply_patch_output, exit_code = runtime.apply_patch(patch)

        if exit_code != "0":
            print(f"Patch application failed. Exit code: {exit_code}. Output:\n{apply_patch_output}")
            report = {"resolved": False, "patch_exists": True, "patch_successfully_applied": False}
        else:
            print("Patch applied successfully. Running evaluation...")
            reward, test_output = runtime._calculate_reward(get_test_output=True, timeout=args.timeout)
            resolved = bool(reward)    # reward is always 1 or 0
            report = {"resolved": resolved, "patch_exists": True, "patch_successfully_applied": True}

    report_json = json.dumps({args.instance_id: report})
    print(f"Evaluation complete. Report: {report_json}")

    if args.output_dir is None:
        output_dir = pathlib.Path("eval-outputs") / args.instance_id
    else:
        output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    report_path.write_text(report_json)
    print(f"Report written to {report_path}")

    if test_output is not None:
        test_output_path = output_dir / "test_output.txt"
        test_output_path.write_text(test_output)
        print(f"Test output written to {test_output_path}")
