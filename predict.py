"""
Three-model genre prediction — CLI wrapper around src.bot.ensemble.GenreClassifier.

Runs CNN, XGBoost, and the stacking ensemble on a single audio file and
prints all three predictions. For the interactive Telegram bot, see
src/bot/telegram_bot.py instead — this script is for one-off / scripted use
(e.g. testing a checkpoint, or a batch job).

Usage:
    python predict.py --audio track.mp3
    python predict.py --audio track.mp3 --top-k 5 --json

Model artifact locations default to env vars (see src/bot/ensemble.py):
    CNN_CHECKPOINT     (default: checkpoints/cnn_model.pt)
    XGB_CHECKPOINT_DIR (default: checkpoints/xgboost)
    STACKER_PATH        (default: checkpoints/stacker.json)
    PREPROCESSOR_DIR    (default: checkpoints/preprocessor)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.bot.ensemble import GenreClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict genre with all three models")
    parser.add_argument("--audio", required=True, help="Path to an audio file")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    args = parser.parse_args()

    print("Loading models...", file=sys.stderr)
    classifier = GenreClassifier.load()

    result = classifier.predict(args.audio, top_k=args.top_k)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"\n{result.filename}")
    print("-" * 45)
    for model_name, pred in [("CNN", result.cnn), ("XGBoost", result.xgb), ("Ensemble", result.stacker)]:
        print(f"\n  {model_name}")
        for rank, (genre, conf) in enumerate(pred.top3, 1):
            print(f"    {rank}. {genre:<20s}  {conf:.1%}")
    print()


if __name__ == "__main__":
    main()