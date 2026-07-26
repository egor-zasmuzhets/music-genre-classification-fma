# Genre classification — CNN + XGBoost + stacker ensemble, plus the Telegram bot.
# CPU-only, single image serves two entrypoints (see bottom).
#
# Build:
#   docker build -t music-genre-bot .
#
# Run the Telegram bot (default):
#   docker run --rm \
#     -e TELEGRAM_BOT_TOKEN=xxxxx \
#     -v $(pwd)/checkpoints:/app/checkpoints:ro \
#     music-genre-bot
#
# Run one-off CLI prediction instead (overrides the default CMD):
#   docker run --rm \
#     -v $(pwd)/checkpoints:/app/checkpoints:ro \
#     -v $(pwd)/samples:/app/samples:ro \
#     music-genre-bot \
#     python predict.py --audio samples/track.mp3

FROM python:3.11-slim

# libsndfile1 — audio decoding for librosa/soundfile.
# ffmpeg          — mp3 support for librosa, and required by pydub for m4a/mp4.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-inference.txt


COPY src/__init__.py src/
COPY src/models/__init__.py src/models/cnn_audio.py src/models/
COPY src/data/__init__.py src/data/preprocessor.py src/data/
COPY src/bot/ src/bot/
COPY predict.py .

# Model artifacts are NOT baked into the image — mount them at runtime via
# -v (see run commands above). Expected layout inside checkpoints/:
#   checkpoints/cnn_model.pt
#   checkpoints/xgboost/*.json
#   checkpoints/stacker.json
#   checkpoints/preprocessor/{scaler,label_encoder,config}.json
# Override any of these with CNN_CHECKPOINT / XGB_CHECKPOINT_DIR /
# STACKER_PATH / PREPROCESSOR_DIR env vars if your layout differs.

CMD ["python", "src/bot/telegram_bot.py"]