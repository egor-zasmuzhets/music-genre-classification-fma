"""
src/data/loader.py
Загрузка сырых данных из CSV файлов FMA
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from src.data.config import paths


class FMALoader:
    """
    Загрузчик метаданных FMA из CSV файлов.

    Особенности:
    - tracks.csv имеет мультииндекс (2 уровня заголовков)
    - features.csv имеет 518 предвычисленных признаков
    - genres.csv содержит иерархию жанров
    """

    def __init__(self, metadata_dir: Optional[Path] = None):
        """
        Args:
            metadata_dir: путь к папке с метаданными (по умолчанию из config)
        """
        self.metadata_dir = metadata_dir or paths.metadata_dir
        self._tracks = None
        self._features = None
        self._genres = None

    @property
    def tracks(self) -> pd.DataFrame:
        """Загружает tracks.csv с мультииндексом"""
        if self._tracks is None:
            tracks_path = self.metadata_dir / "tracks.csv"
            if not tracks_path.exists():
                raise FileNotFoundError(f"tracks.csv не найден: {tracks_path}")

            self._tracks = pd.read_csv(
                tracks_path,
                header=[0, 1],
                index_col=0
            )
            print(f"Загружено tracks: {self._tracks.shape}")
        return self._tracks

    @property
    def features(self) -> pd.DataFrame:
        """Загружает features.csv (518 признаков)"""
        if self._features is None:
            features_path = self.metadata_dir / "features.csv"
            if not features_path.exists():
                raise FileNotFoundError(f"features.csv не найден: {features_path}")

            self._features = pd.read_csv(
                features_path,
                index_col=0
            )
            print(f"Загружено features: {self._features.shape}")
        return self._features

    @property
    def genres(self) -> pd.DataFrame:
        """Загружает genres.csv (иерархия жанров)"""
        if self._genres is None:
            genres_path = self.metadata_dir / "genres.csv"
            if not genres_path.exists():
                raise FileNotFoundError(f"genres.csv не найден: {genres_path}")

            self._genres = pd.read_csv(
                genres_path,
                index_col=0
            )
            print(f"Загружено genres: {self._genres.shape}")
        return self._genres

    def get_tracks_by_subset(self, subset: str = "medium") -> pd.DataFrame:
        """
        Фильтрует треки по подмножеству (small/medium/large)

        Args:
            subset: 'small', 'medium', или 'large'

        Returns:
            DataFrame с отфильтрованными треками
        """
        tracks = self.tracks

        # Колонка с подмножеством может быть в разных местах
        if ('set', 'subset') in tracks.columns:
            filtered = tracks[tracks[('set', 'subset')] == subset].copy()
        else:
            # Fallback: ищем колонку с 'subset' на любом уровне
            subset_cols = [c for c in tracks.columns if 'subset' in c]
            if subset_cols:
                filtered = tracks[tracks[subset_cols[0]] == subset].copy()
            else:
                raise KeyError("Не найдена колонка с информацией о подмножестве")

        print(f"Треков в {subset}: {len(filtered)}")
        return filtered

    def get_available_splits(self, tracks_df: pd.DataFrame) -> Dict[str, pd.Index]:
        """
        Возвращает индексы треков для train/val/test по официальному разбиению

        Args:
            tracks_df: DataFrame с колонкой ('set', 'split')

        Returns:
            Словарь с индексами для 'training', 'validation', 'test'
        """
        if ('set', 'split') not in tracks_df.columns:
            raise KeyError("В данных нет колонки с разбиением ('set', 'split')")

        splits = {
            'training': tracks_df[tracks_df[('set', 'split')] == 'training'].index,
            'validation': tracks_df[tracks_df[('set', 'split')] == 'validation'].index,
            'test': tracks_df[tracks_df[('set', 'split')] == 'test'].index
        }

        # Проверяем, что все треки покрыты
        total = sum(len(v) for v in splits.values())
        if total != len(tracks_df):
            print(f"Предупреждение: {len(tracks_df) - total} треков без разметки")

        return splits

    def get_genre_mapping(self) -> Dict[int, str]:
        """Возвращает словарь {genre_id: genre_name}"""
        return self.genres['title'].to_dict()

    def print_info(self):
        """Выводит информацию о загруженных данных"""
        print("=" * 50)
        print("FMALoader Info")
        print("=" * 50)
        print(f"Metadata directory: {self.metadata_dir}")
        print(f"Tracks: {self.tracks.shape}")
        print(f"Features: {self.features.shape}")
        print(f"Genres: {self.genres.shape}")
        print(f"Subsets available: {self.tracks[('set', 'subset')].unique()}")
        print(f"Splits available: {self.tracks[('set', 'split')].unique()}")