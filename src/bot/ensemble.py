"""
GenreClassifier — единая точка инференса для Telegram-бота.

Загружает все три модели один раз при старте, принимает путь к MP3,
возвращает структурированный результат.

Использование:
    from genre_classifier import GenreClassifier, ClassifierResult

    classifier = GenreClassifier.load()
    result = classifier.predict("track.mp3")

    print(result.stacker.top1_genre)   # "Rock"
    print(result.stacker.top1_conf)    # 0.61
    print(result.to_dict())            # для JSON-ответа
"""

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy import stats

from src.models.cnn_audio import AudioCNN
from src.data.preprocessor import DataPreprocessor
from src.utils.config import paths

# ═══════════════════════════════════════════════════════════
CNN_CHECKPOINT_PATH = Path("")
XGB_CHECKPOINT_DIR  = Path("")
STACKER_PATH        = Path("")
# ═══════════════════════════════════════════════════════════

_DATASET_ID    = "medium_10"
_N_MFCC        = 40
_TARGET_FRAMES = 430
_SR_CNN        = 22050
_N_FFT         = 2048
_HOP_LENGTH    = 512
_WIN_LENGTH    = 2048


# ── результаты ────────────────────────────────────────────

@dataclass
class ModelPrediction:
    top1_genre: str
    top1_conf:  float
    top3:       List[Tuple[str, float]]   # [(genre, conf), ...]
    all_proba:  List[float]               # все вероятности в порядке genre_names

    def to_dict(self) -> dict:
        return {
            "top1_genre": self.top1_genre,
            "top1_conf":  round(self.top1_conf, 4),
            "top3": [{"genre": g, "conf": round(c, 4)} for g, c in self.top3],
        }


@dataclass
class ClassifierResult:
    filename: str
    cnn:      ModelPrediction
    xgb:      ModelPrediction
    stacker:  ModelPrediction
    genre_names: List[str] = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "cnn":     self.cnn.to_dict(),
            "xgb":     self.xgb.to_dict(),
            "stacker": self.stacker.to_dict(),
        }


# ── основной класс ────────────────────────────────────────

class GenreClassifier:
    """
    Загружает CNN, XGBoost и Stacker один раз, переиспользует для любого MP3.

    Attributes:
        genre_names: Список жанров в порядке кодирования меток.
        device:      Torch-устройство ('cuda' / 'cpu').
    """

    def __init__(
        self,
        cnn_model:  AudioCNN,
        xgb_model:  xgb.XGBClassifier,
        stacker:    xgb.XGBClassifier,
        scaler,
        genre_names: List[str],
        device: str,
    ) -> None:
        self._cnn     = cnn_model
        self._xgb     = xgb_model
        self._stacker = stacker
        self._scaler  = scaler
        self.genre_names = genre_names
        self.device = device


    @classmethod
    def load(
        cls,
        cnn_checkpoint: Path = CNN_CHECKPOINT_PATH,
        xgb_dir:        Path = XGB_CHECKPOINT_DIR,
        stacker_path:   Path = STACKER_PATH,
        device:         Optional[str] = None,
    ) -> "GenreClassifier":
        """Загружает все модели и возвращает готовый классификатор."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        prep_dir = paths.fma_features_dataset_dir / _DATASET_ID / "preprocessor"
        preprocessor = DataPreprocessor()
        preprocessor.load(prep_dir)

        cnn = AudioCNN.load(cnn_checkpoint, device=device)
        cnn.eval()

        json_files = sorted(xgb_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"*.json не найден в {xgb_dir}")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(json_files[0]))

        stacker = xgb.XGBClassifier()
        stacker.load_model(str(stacker_path))

        return cls(
            cnn_model=cnn,
            xgb_model=xgb_model,
            stacker=stacker,
            scaler=preprocessor.scaler,
            genre_names=preprocessor.class_names,
            device=device,
        )


    @staticmethod
    def _to_wav_if_needed(audio_path: Path) -> Path:
        """
        Конвертирует m4a/mp4 в wav через pydub.
        Для остальных форматов возвращает путь без изменений.
        """
        if audio_path.suffix.lower() not in {".m4a", ".mp4"}:
            return audio_path
        from pydub import AudioSegment
        wav_path = audio_path.with_suffix(".wav")
        fmt = audio_path.suffix.lstrip(".")
        AudioSegment.from_file(str(audio_path), format=fmt).export(str(wav_path), format="wav")
        return wav_path


    def predict(self, mp3_path: str | Path, top_k: int = 3) -> ClassifierResult:
        """
        Классифицирует жанр MP3-файла всеми тремя моделями.

        Args:
            mp3_path: Путь к MP3-файлу.
            top_k:    Размер топ-N в результате (по умолч. 3).

        Returns:
            ClassifierResult с предсказаниями CNN, XGBoost и Stacker.

        Raises:
            FileNotFoundError: Если файл не существует.
            RuntimeError: Если не удалось извлечь признаки.
        """
        mp3_path = Path(mp3_path)
        if not mp3_path.exists():
            raise FileNotFoundError(mp3_path)

        audio_path = self._to_wav_if_needed(mp3_path)
        cnn_proba = self._predict_cnn(audio_path)
        xgb_proba = self._predict_xgb(audio_path)

        stack_input   = np.concatenate([cnn_proba, xgb_proba]).reshape(1, -1)
        stacker_proba = self._stacker.predict_proba(stack_input)[0]

        return ClassifierResult(
            filename    = mp3_path.name,
            cnn         = self._make_prediction(cnn_proba,     top_k),
            xgb         = self._make_prediction(xgb_proba,     top_k),
            stacker     = self._make_prediction(stacker_proba, top_k),
            genre_names = self.genre_names,
        )

    # ── внутренние методы ─────────────────────────────────

    def _make_prediction(self, proba: np.ndarray, top_k: int) -> ModelPrediction:
        top_idx = np.argsort(proba)[::-1][:top_k]
        return ModelPrediction(
            top1_genre = self.genre_names[top_idx[0]],
            top1_conf  = float(proba[top_idx[0]]),
            top3       = [(self.genre_names[i], float(proba[i])) for i in top_idx],
            all_proba  = proba.tolist(),
        )

    @torch.no_grad()
    def _predict_cnn(self, mp3_path: Path) -> np.ndarray:
        audio, _ = librosa.load(str(mp3_path), sr=_SR_CNN, mono=True)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=_SR_CNN, n_mfcc=_N_MFCC,
            n_fft=_N_FFT, hop_length=_HOP_LENGTH, win_length=_WIN_LENGTH,
        ).astype(np.float32)

        n = mfcc.shape[1]
        if n <= _TARGET_FRAMES:
            mfcc = np.pad(mfcc, ((0, 0), (0, _TARGET_FRAMES - n)))
        else:
            start = (n - _TARGET_FRAMES) // 2
            mfcc = mfcc[:, start: start + _TARGET_FRAMES]

        delta  = np.diff(mfcc,  axis=1, prepend=mfcc[:, :1])
        delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
        result = np.stack([mfcc, delta, delta2], axis=0).astype(np.float32)
        mean = result.mean(axis=(1, 2), keepdims=True)
        std  = result.std(axis=(1, 2),  keepdims=True) + 1e-8
        result = (result - mean) / std

        x = torch.tensor(result).unsqueeze(0).to(self.device)
        return torch.softmax(self._cnn(x), dim=1).cpu().numpy()[0]

    def _predict_xgb(self, mp3_path: Path) -> np.ndarray:
        X = self._extract_features_xgb(mp3_path)
        return self._xgb.predict_proba(self._scaler.transform(X))[0]

    @staticmethod
    def _extract_features_xgb(mp3_path: Path) -> np.ndarray:
        feature_sizes = dict(
            chroma_stft=12, chroma_cqt=12, chroma_cens=12,
            tonnetz=6, mfcc=20, rmse=1, zcr=1,
            spectral_centroid=1, spectral_bandwidth=1,
            spectral_contrast=7, spectral_rolloff=1,
        )
        moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')
        cols = []
        for name, size in feature_sizes.items():
            for moment in moments:
                cols.extend((name, moment, f'{i+1:02d}') for i in range(size))
        columns = pd.MultiIndex.from_tuples(
            cols, names=('feature', 'statistics', 'number')).sort_values()

        features = pd.Series(index=columns, dtype=np.float32)

        def feature_stats(name, values):
            features.loc[(name, 'mean')]     = np.mean(values, axis=1).astype(np.float32)
            features.loc[(name, 'std')]      = np.std(values, axis=1).astype(np.float32)
            features.loc[(name, 'skew')]     = stats.skew(values, axis=1).astype(np.float32)
            features.loc[(name, 'kurtosis')] = stats.kurtosis(values, axis=1).astype(np.float32)
            features.loc[(name, 'median')]   = np.median(values, axis=1).astype(np.float32)
            features.loc[(name, 'min')]      = np.min(values, axis=1).astype(np.float32)
            features.loc[(name, 'max')]      = np.max(values, axis=1).astype(np.float32)

        warnings.filterwarnings('error', module='librosa')
        x, sr = librosa.load(str(mp3_path), sr=None, mono=True)

        feature_stats('zcr', librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=84, tuning=None))
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        feature_stats('tonnetz', librosa.feature.tonnetz(chroma=f))
        del cqt

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        del x

        feature_stats('chroma_stft', librosa.feature.chroma_stft(S=stft**2, n_chroma=12))
        feature_stats('rmse',        librosa.feature.rms(S=stft))
        feature_stats('spectral_centroid',  librosa.feature.spectral_centroid(S=stft))
        feature_stats('spectral_bandwidth', librosa.feature.spectral_bandwidth(S=stft))
        feature_stats('spectral_contrast',  librosa.feature.spectral_contrast(S=stft, n_bands=6))
        feature_stats('spectral_rolloff',   librosa.feature.spectral_rolloff(S=stft))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        feature_stats('mfcc', librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))

        return features.values.reshape(1, -1)


# ── быстрая проверка ──────────────────────────────────────

if __name__ == "__main__":
    import sys
    mp3 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/path/to/track.mp3")

    classifier = GenreClassifier.load()
    result = classifier.predict(mp3)

    print(f"\n{result.filename}")
    print("─" * 45)
    for model_name, pred in [("CNN", result.cnn), ("XGBoost", result.xgb), ("Stacker", result.stacker)]:
        print(f"\n  {model_name}")
        for rank, (genre, conf) in enumerate(pred.top3, 1):
            print(f"    {rank}. {genre:<20s}  {conf:.1%}")
    print()