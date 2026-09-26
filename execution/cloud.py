import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["core", "benchmarks", "diversity", "attacks", "textseal"],
        required=True,
    )
    parser.add_argument("--gpu")
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--memory", type=int, default=32768)
    parser.add_argument("--module", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.module.startswith("experiments."):
        parser.error("--module must name an experiment module")
    import modal

    root = Path(__file__).resolve().parents[1]
    profiles = json.loads(Path(__file__).with_name("environments.json").read_text())
    profile = profiles[args.profile]
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "build-essential")
        .pip_install(*profile["packages"])
    )
    for command in profile.get("commands", []):
        image = image.run_commands(command)
    image = image.env(
        {
            "PYTHONPATH": "/workspace:/kth",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "NLTK_DATA": "/nltk_data",
        }
    )
    image = image.add_local_dir(
        str(root),
        "/workspace",
        ignore=[".git", ".venv", "__pycache__", "runs", "models"],
    )
    cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
    data = modal.Volume.from_name("prc-data", create_if_missing=False)
    results = modal.Volume.from_name("prc-paper-results", create_if_missing=True)
    app = modal.App("prc-paper-experiments")

    @app.function(
        image=image,
        gpu=args.gpu,
        cpu=4,
        memory=args.memory,
        timeout=args.timeout,
        retries=0,
        volumes={"/cache": cache, "/data": data, "/results": results},
    )
    def execute(module, arguments):
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", module, *arguments], cwd="/workspace", check=True
        )
        results.commit()

    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    with app.run():
        execute.remote(args.module, arguments)


if __name__ == "__main__":
    main()
