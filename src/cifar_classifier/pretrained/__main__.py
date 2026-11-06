"""CLI: python -m cifar_classifier.pretrained {train,evaluate,plot-history}."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import EfficientNetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cifar_classifier.pretrained")
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train", help="Fine-tune pretrained EfficientNetV2-B0")
    train_cmd.add_argument("--data-dir", type=Path, default=Path("cifar-100"))
    train_cmd.add_argument("--run-dir", type=Path)
    train_cmd.add_argument("--epochs", type=int, default=30)
    train_cmd.add_argument("--batch-size", type=int, default=32)
    train_cmd.add_argument("--image-size", type=int, default=224)
    train_cmd.add_argument("--patience", type=int, default=4)
    train_cmd.add_argument("--num-workers", type=int, default=0)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--validation-size", type=int, default=5_000)
    train_cmd.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    train_cmd.add_argument("--no-pretrained", action="store_true",
                           help="Initialize randomly for offline pipeline checks")
    train_cmd.add_argument("--pretrained-weights", type=Path,
                           help="Local ImageNet weights for tf_efficientnetv2_b0.in1k")
    train_cmd.add_argument("--cache-dir", type=Path,
                           default=Path("artifacts/pretrained/cache"),
                           help="Cache downloaded ImageNet weights")
    train_cmd.add_argument("--limit-train", type=int, help="Use only N training samples")
    train_cmd.add_argument("--limit-val", type=int, help="Use only N validation samples")

    eval_cmd = commands.add_parser("evaluate", help="Evaluate a saved .pth checkpoint")
    eval_cmd.add_argument("--checkpoint", type=Path, required=True)
    eval_cmd.add_argument("--data-dir", type=Path, default=Path("cifar-100"))
    eval_cmd.add_argument("--output-dir", type=Path)
    eval_cmd.add_argument("--batch-size", type=int)
    eval_cmd.add_argument("--image-size", type=int,
                          help="Only needed for a raw notebook checkpoint")
    eval_cmd.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    eval_cmd.add_argument("--limit-test", type=int, help="Use only N test samples for checks")

    plot_cmd = commands.add_parser("plot-history", help="Save training curves from history.csv")
    plot_cmd.add_argument("--history", type=Path, required=True)
    plot_cmd.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from .train import train

        config = EfficientNetConfig(
            epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size,
            patience=args.patience, num_workers=args.num_workers, seed=args.seed,
            validation_size=args.validation_size,
        )
        if args.limit_train is not None and args.limit_train < 1:
            raise ValueError("--limit-train must be positive")
        if args.limit_val is not None and args.limit_val < 1:
            raise ValueError("--limit-val must be positive")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.run_dir or Path("artifacts/pretrained/runs") / f"efficientnetv2-b0-{timestamp}"
        result = train(
            args.data_dir, run_dir, config, device_name=args.device,
            pretrained=not args.no_pretrained, limit_train=args.limit_train,
            limit_val=args.limit_val, pretrained_weights=args.pretrained_weights,
            cache_dir=args.cache_dir,
        )
        print(json.dumps({"run_dir": str(run_dir), "best_epoch": result["best_epoch"],
                          "best_val_loss": result["best_val_loss"]}, indent=2))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        if args.limit_test is not None and args.limit_test < 1:
            raise ValueError("--limit-test must be positive")
        output_dir = args.output_dir or args.checkpoint.parent / "test-evaluation"
        result = evaluate(
            args.checkpoint, args.data_dir, output_dir, device_name=args.device,
            batch_size=args.batch_size, image_size=args.image_size, limit_test=args.limit_test,
        )
        print(json.dumps(result, indent=2))
        print(f"Reports: {output_dir}")
    else:
        from .plots import plot_history

        output = args.output or args.history.with_name("training_history.png")
        print(plot_history(args.history, output))


if __name__ == "__main__":
    main()
