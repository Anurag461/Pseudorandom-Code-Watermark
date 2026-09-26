import argparse
from pathlib import Path
from baselines.wang import analysis, lm, prepare
from baselines.wang.config import TEMPERATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "generate", "analyze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--temperature", type=float, choices=TEMPERATURES)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare.prepare(args.output, {})
    elif args.stage == "generate":
        if args.cache is None or args.temperature is None:
            parser.error("generate requires --cache and --temperature")
        lm.run_temperature(args.output, args.temperature, args.cache, {})
    else:
        analysis.run(args.output, {})


if __name__ == "__main__":
    main()
