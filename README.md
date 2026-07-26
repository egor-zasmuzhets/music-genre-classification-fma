# Music Genre Classification (FMA-Medium)

Classifying tracks into one of 16 genres using two contrasting approaches — a
gradient-boosted baseline on aggregated audio statistics, and a convolutional
network trained directly on MFCC time-frequency maps. Built as a Bachelor's
thesis at Belarusian State University (Mechanics and Mathematics Faculty);
the full write-up (54 pages, methodology, math derivations, error analysis)
is available on request.

## Problem

Automatic genre tagging matters at streaming scale, where manual tagging of
millions of tracks is infeasible. It's also a genuinely hard ML problem:
genre boundaries are subjective (inter-rater agreement caps achievable
accuracy), classes are severely imbalanced, and the dataset used here —
[FMA-Medium](https://github.com/mdeff/fma) — reflects that reality rather
than a cleaned-up benchmark like GTZAN.

- **25,000 tracks**, 16 root genres, 30s clips, licensed under Creative Commons
- **Class imbalance ratio > 100:1** — Rock/Electronic: ~5,000 tracks each;
  Blues/Easy Listening: under 60
- Split 80/10/10 (train/val/test), stratified so **no artist or label leaks
  across splits** — otherwise a model can memorize artist fingerprints
  instead of learning genre

## Approach

Two models representing two different paradigms, trained and compared on
identical data:

| | XGBoost (baseline) | AudioCNN (main) |
|---|---|---|
| Input | 518 aggregated statistics (`features.csv`) | 3×40×430 MFCC map (MFCC + Δ + ΔΔ) |
| Temporal info | Discarded (mean/std/skew/kurtosis over 30s) | Preserved (2D conv over time × frequency) |
| Why | Fast, interpretable, strong with little data | Captures rhythm/structure classical stats can't |

The baseline uses the dataset's precomputed `features.csv` (chroma, MFCC,
spectral contrast, tonnetz — 518 dims, see thesis §2.1.5) so a first quality
signal is available before any custom feature engineering. The CNN operates
on raw MFCC maps extracted with `librosa` (`n_mfcc=40, n_fft=2048,
hop_length=512`, center-cropped to 430 frames ≈ 10s).

## Results

### Final comparison (test set)

| Metric | XGBoost | AudioCNN |
|---|---|---|
| Accuracy | **0.624** | 0.604 |
| Macro F1 | 0.414 | **0.468** |
| Weighted F1 | **0.600** | 0.595 |
| Top-3 Accuracy | 0.837 | **0.854** |

XGBoost wins on raw accuracy (it defaults to majority classes more safely);
AudioCNN wins on Macro F1 and Top-3 — i.e. it's meaningfully better at rare
genres and more often ranks the correct genre in its top guesses, even when
its single top prediction is wrong. Macro F1 is the metric that matters most
here given the 100:1 class imbalance (see thesis §2.3 for why raw accuracy
is misleading under imbalance).

### By rarity group (AudioCNN gain over XGBoost)

| Group | XGBoost F1 | AudioCNN F1 | Gain |
|---|---|---|---|
| Very rare (7 genres, <100 examples) | 0.285 | 0.446 | **+0.161** |
| Rare (7 genres, 100–500) | 0.393 | 0.408 | +0.015 |
| Medium (2 genres, 500–1400) | 0.756 | 0.757 | +0.001 |

The CNN's entire advantage is concentrated in the very-rare bucket. On
common genres the two models are statistically indistinguishable — the
architecture doesn't matter once there's enough data.

### The interesting part: the CNN collapsed first, and had to be recovered

Naive training (weighted sampler + MixUp + basic SpecAugment) didn't work —
by epoch 18 the CNN had **collapsed into predicting the 4 majority classes**,
scoring Macro F1 0.313 (worse than baseline) while individual F1 for Blues,
Easy Listening, and Soul-RnB was exactly 0. Frequency Masking was actively
destroying the sparse spectral signal that separates rare genres, and
WeightedRandomSampler/MixUp weren't strong enough against a 437:1 class
ratio.

Recovery required four coordinated countermeasures over 30 further epochs:

1. **Progressive class reintroduction** — majority classes (Electronic,
   Rock) temporarily excluded from training, reintroduced in 4 stages
   (0% → 20% → 60% → 100%) as the model regained footing on rare classes.
2. **Rarity-aware augmentation** — Frequency Masking replaced with
   Frequency Warping (preserves full spectral content); augmentation
   intensity and crop count scaled inversely with class frequency
   (very-rare classes: up to 4 crops/track, 0.7 masking probability).
3. **Curriculum augmentation** — transformation intensity ramped from 0.3×
   to 1.0× within each stage, so the network learns clean patterns first.
4. **Progressive layer unfreezing** — conv layers thawed one at a time
   across 4 stages (`conv4-5` → `conv3` → `conv2` → all), each with lower
   LR and stronger regularization (dropout 0.10 → 0.50, weight decay
   1e-4 → 2e-3).

Result: Macro F1 recovered from 0.313 to **0.468**, surpassing the XGBoost
baseline, with the largest gains exactly where it mattered — genres with
under 100 training examples.

Full per-class breakdown, confusion matrices, and error analysis (e.g. why
`Experimental ↔ Electronic` is the model's single biggest confusion pair) are
in the thesis, §3.3–3.5.

## Project structure

```
├── configsexample/       # Template configs — copy to configs/ and fill in (see Setup)
├── notebooks/            # Exploration, XGBoost baseline, full CNN pipeline
├── src/
│   ├── data/              # Audio loading, MFCC extraction, caching, torch Dataset
│   ├── models/            # AudioCNN, XGBoost wrapper
│   ├── training/          # Training loops, grid search, metrics tracking
│   ├── bot/                # Telegram bot + stacking ensemble for live inference
│   └── utils/              # Config loading, logging
├── tests/                 # Unit tests for the data pipeline
├── predict.py             # One-off CLI: all 3 models (see Inference below)
├── Dockerfile             # CPU image — runs the bot by default, or predict.py
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/egor-zasmuzhets/music-genre-classification-fma
cd music-genre-classification-fma
pip install -r requirements.txt

# Config is environment-specific (local dataset paths) and not committed.
# Copy the template and fill in your paths:
cp -r configsexample configs
# edit configs/paths.yaml with your FMA zip locations
# edit configs/models.yaml with hyperparameters (see thesis §3.3.1 / §3.4.1
# for the exact values used to produce the results above)
```

Download [FMA-Medium](https://github.com/mdeff/fma) and point
`configs/paths.yaml` at the zip archive and metadata directory.

## Inference

Production inference doesn't run the CNN alone — it runs all three models
and combines them with a stacking ensemble (`src/bot/ensemble.py`):

1. **AudioCNN** predicts from the raw MFCC map.
2. **XGBoost** predicts from 518 aggregated statistics (same features as
   the baseline, extracted live from the audio).
3. A **stacker** (a third XGBoost model, trained on the concatenated
   32-dim probability vector from CNN + XGBoost) combines both into a
   final prediction — this is what the Telegram bot shows as "Ensemble".

Model artifacts (checkpoint, XGBoost model, stacker, and the fitted
preprocessor — scaler + label encoder, see `src/data/preprocessor.py`)
aren't committed to git and need to be supplied separately, laid out as:

```
checkpoints/
├── cnn_model.pt
├── xgboost/*.json
├── stacker.json
└── preprocessor/{scaler,label_encoder,config}.json
```

(Paths are overridable via `CNN_CHECKPOINT`, `XGB_CHECKPOINT_DIR`,
`STACKER_PATH`, `PREPROCESSOR_DIR` env vars — see `src/bot/ensemble.py`.)

### One-off CLI prediction

```bash
pip install -r requirements-inference.txt
python predict.py --audio track.mp3
```

```
track.mp3
---------------------------------------------

  CNN
    1. Rock                   74.2%
    2. Experimental            9.1%
    3. Folk                    6.3%

  XGBoost
    1. Rock                   61.0%
    ...

  Ensemble
    1. Rock                   79.5%
    ...
```

### Telegram bot

`src/bot/telegram_bot.py` wraps the same ensemble in an interactive bot —
send it an audio file (mp3/wav/flac/ogg/m4a/mp4/voice message, up to 20MB)
and it replies with per-model predictions, a probability bar chart, and
per-model timing. It also supports searching and classifying tracks
directly via the Deezer API (`/search <query>`), without needing a local
file at all.

```bash
pip install -r requirements-inference.txt
export TELEGRAM_BOT_TOKEN=xxxxx
python src/bot/telegram_bot.py
```

### Docker

One image, two modes — bot by default, CLI prediction by overriding `CMD`:

```bash
docker build -t music-genre-bot .

# Bot (long-running)
docker run --rm \
  -e TELEGRAM_BOT_TOKEN=xxxxx \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  music-genre-bot

# One-off CLI prediction
docker run --rm \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/samples:/app/samples:ro \
  music-genre-bot \
  python predict.py --audio samples/track.mp3
```

The image only copies the inference-relevant slice of `src/` (models,
preprocessor, bot) — not notebooks, training code, or tests — and is
independent of `configs/` by design: `src/bot/ensemble.py` resolves all
four artifact paths explicitly (env vars, see above) rather than through
`src.utils.config`, which eagerly reads `configs/paths.yaml` at import
time and isn't something a deployment container should depend on.

## Tech stack

Python, PyTorch, XGBoost, librosa, scikit-learn, pandas · Docker for inference
· Telegram bot for live demo (`src/bot/telegram_bot.py`, stacking ensemble
over both models)

## References

Built on the [FMA dataset](https://github.com/mdeff/fma) (Defferrard et al.,
ISMIR 2017). Full literature review and comparison against transfer-learning
approaches (Whisper embeddings, CT-GateNet, etc.) in thesis §1.3.3.