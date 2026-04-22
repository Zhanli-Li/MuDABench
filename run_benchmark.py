import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run this repository's agent + agent-mode evaluation end-to-end")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-concurrency", type=int, default=1)
    parser.add_argument("--doc-concurrency", type=int, default=1)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--backend", default="chatdoc")
    parser.add_argument("--backend-entrypoint", default="")
    parser.add_argument("--agent-log-dir", default="")
    parser.add_argument("--metadata-description", default=(
        "Field 1: symbol, stock code. Field 2: year, year. Field 3: doctype, document type."
    ))
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_root, exist_ok=True)

    agent_output = os.path.join(args.output_root, "agent_output.json")
    eval_dir = os.path.join(args.output_root, "eval")

    cmd_agent = [
        sys.executable,
        os.path.join(this_dir, "agent", "agent_runner.py"),
        "--dataset",
        args.dataset,
        "--output",
        agent_output,
        "--metadata-description",
        args.metadata_description,
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
        "--question-concurrency",
        str(args.question_concurrency),
        "--doc-concurrency",
        str(args.doc_concurrency),
        "--backend",
        args.backend,
    ]
    if args.backend_entrypoint:
        cmd_agent += ["--backend-entrypoint", args.backend_entrypoint]
    if args.agent_log_dir:
        cmd_agent += ["--agent-log-dir", args.agent_log_dir]

    cmd_eval = [
        sys.executable,
        os.path.join(this_dir, "eval", "evaluate.py"),
        "--dataset",
        args.dataset,
        "--agent-output",
        agent_output,
        "--output-dir",
        eval_dir,
        "--eval-mode",
        "agent",
        "--judge-concurrency",
        str(args.judge_concurrency),
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
    ]

    print("running:", " ".join(cmd_agent))
    subprocess.run(cmd_agent, check=True)

    print("running:", " ".join(cmd_eval))
    subprocess.run(cmd_eval, check=True)

    print("done")
    print("agent output:", agent_output)
    print("eval summary:", os.path.join(eval_dir, "eval_summary.json"))
    print("eval log:", os.path.join(eval_dir, "eval_log.json"))


if __name__ == "__main__":
    main()
