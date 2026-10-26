"""Command line entry point for the CIFAR-100 WRN project."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import LEGACY_CHECKPOINT, LEGACY_HISTORIES, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cifar-classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    train_cmd = commands.add_parser("train", help="Train WRN-16-4 from scratch")
    train_cmd.add_argument("--data-dir", type=Path, default=Path("cifar-100"))
    train_cmd.add_argument("--run-dir", type=Path)
    train_cmd.add_argument("--epochs", type=int, default=200)
    train_cmd.add_argument("--batch-size", type=int, default=64)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--validation-size", type=int, default=5_000)
    train_cmd.add_argument("--schedule", choices=("plateau", "cosine"), default="plateau")

    eval_cmd = commands.add_parser("evaluate", help="Evaluate a checkpoint on CIFAR-100 test")
    eval_cmd.add_argument("--data-dir", type=Path, default=Path("cifar-100"))
    eval_cmd.add_argument("--checkpoint", type=Path,
                          default=Path("checkpoints") / LEGACY_CHECKPOINT)
    eval_cmd.add_argument("--output-dir", type=Path,
                          default=Path("artifacts/evaluation/legacy-wrn200"))
    eval_cmd.add_argument("--batch-size", type=int, default=64)
    eval_cmd.add_argument("--seed", type=int, default=42)
    eval_cmd.add_argument("--validation-size", type=int, default=5_000)

    plot_cmd = commands.add_parser("plot-history", help="Save one figure per history CSV")
    plot_cmd.add_argument("--history", type=Path, action="append",
                          help="CSV path; repeat to draw several runs")
    plot_cmd.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    plot_cmd.add_argument("--monitor", choices=("val_accuracy", "val_loss"),
                          help="Checkpoint selection metric for custom CSV files")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from .train import train

        config = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            validation_size=args.validation_size, schedule=args.schedule,
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.run_dir or Path("artifacts/runs") / f"wrn16_4-{timestamp}"
        result = train(args.data_dir, run_dir, config)
        print(json.dumps({"run_dir": str(run_dir), **result["best_validation"]}, indent=2))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(
            args.checkpoint, args.data_dir, args.output_dir,
            batch_size=args.batch_size, seed=args.seed,
            validation_size=args.validation_size,
        )
        print(json.dumps(result, indent=2))
        print(f"Reports saved in {args.output_dir}")
    else:
        from .plots import plot_history

        histories = args.history or [Path("checkpoints") / name for name in LEGACY_HISTORIES]
        for history_path in histories:
            output = plot_history(history_path, args.output_dir, monitor=args.monitor)
            print(output)


if __name__ == "__main__":
    main()
