# Data Module — работа с датасетом FMA

Этот модуль отвечает за загрузку, предобработку и кэширование данных из датасета [FMA (Free Music Archive)](https://github.com/mdeff/fma).

## 📁 Структура модуля

```
src/data/
├── config.py           # Конфигурация путей и параметров
├── loader.py           # Загрузка сырых CSV файлов FMA
├── preprocessor.py     # Предобработка: кодирование, нормализация, фильтрация
├── pipeline.py         # Полный пайплайн подготовки данных
├── load_processed.py   # Быстрая загрузка уже обработанных данных
└── README.md           # Эта документация
```

---

## 🎯 Основные компоненты

### 1. `config.py` — управление конфигурацией

**Что делает:**
- Определяет корень проекта (поиск `configs/paths.yaml`)
- Загружает YAML конфиги
- Предоставляет глобальные объекты `paths` и `audio_params`

**Как использовать:**

```python
from src.data.config import paths, audio_params

# Пути к директориям
print(paths.processed_data_dir)   # ./data/processed
print(paths.models_dir)           # ./models

# Параметры аудио
print(audio_params.sample_rate)   # 22050
print(audio_params.n_mels)        # 128
```

**Ключевые свойства `paths`:**

| Свойство | Назначение |
|----------|------------|
| `metadata_dir` | Директория с `tracks.csv`, `features.csv` |
| `raw_data_dir` | Для сырых аудиофайлов (MP3) |
| `processed_data_dir` | Для кэшированных данных (`.npy`, `.pkl`) |
| `checkpoints_dir` | Для весов моделей |
| `results_dir` | Для графиков и метрик |

---

### 2. `loader.py` — загрузка метаданных FMA

**Что делает:**
- Загружает `tracks.csv` (мультииндекс, 2 уровня)
- Загружает `features.csv` (518 предвычисленных признаков)
- Загружает `genres.csv` (иерархия жанров)
- Фильтрует треки по подмножеству (small/medium/large)
- Извлекает официальное разбиение train/val/test

**Как использовать:**

```python
from src.data.loader import FMALoader

loader = FMALoader()

# Загрузить все треки FMA Medium
tracks = loader.get_tracks_by_subset("medium")

# Получить признаки
features = loader.features  # (106574, 518)

# Получить разбиение
splits = loader.get_available_splits(tracks)
train_idx = splits['training']   # индексы треков для обучения
```

**Особенности:**
- `tracks.csv` имеет мультииндекс: колонки вида `('track', 'genre_top')`
- `features.csv` имеет тройной мультииндекс: `('feature', 'method', 'statistic')`
- Загрузка происходит **лениво** (данные загружаются при первом обращении)

---

### 3. `preprocessor.py` — предобработка данных

**Что делает:**
- Фильтрует редкие жанры (по `min_samples_per_genre`)
- Кодирует текстовые метки в числа (`LabelEncoder`)
- Нормализует признаки (`StandardScaler`)
- Вычисляет веса классов для борьбы с дисбалансом

**Как использовать:**

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(min_samples_per_genre=100)

# 1. Фильтрация редких жанров
tracks_filtered = preprocessor.filter_rare_genres(tracks, ('track', 'genre_top'))

# 2. Кодирование меток (fit на train, transform на val/test)
y_train_enc, y_val_enc, y_test_enc = preprocessor.encode_labels(
    y_train, y_val, y_test
)

# 3. Нормализация признаков
X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
    X_train, X_val, X_test
)

# 4. Веса классов
class_weights = preprocessor.get_class_weights(y_train_enc)
```

**Важные детали:**
- `fit` выполняется **только на тренировочной выборке**
- `transform` применяется к валидационной и тестовой (нет утечки данных)
- Веса классов вычисляются по формуле `balanced`: `weight = n_samples / (n_classes * n_class_samples)`

---

### 4. `pipeline.py` — полный пайплайн

**Что делает:**
Объединяет все шаги в один автоматизированный процесс:

1. Загрузка треков FMA Small/Medium/Large
2. Фильтрация треков без жанра
3. Удаление редких жанров
4. Использование официального разбиения FMA (80/10/10)
5. Сопоставление признаков из `features.csv` с треками
6. Кодирование меток и нормализация признаков
7. Сохранение в кэш (`.npy` + `.pkl`)

**Как использовать:**

```python
from src.data.pipeline import DataPipeline

# Создаём пайплайн для FMA Medium
pipeline = DataPipeline(
    subset="medium",              # small, medium, large
    min_samples_per_genre=100,    # минимально треков на жанр
    use_features=True,            # использовать предвычисленные признаки
)

# Запускаем (при первом запуске — долго, потом — из кэша)
data = pipeline.run()

# Получаем данные
X_train = data['X_train']   # numpy array, нормализованные признаки
X_val   = data['X_val']
X_test  = data['X_test']
y_train = data['y_train']   # закодированные метки (0..num_classes-1)
y_val   = data['y_val']
y_test  = data['y_test']

# Дополнительно
genre_names = data['genre_names']   # список названий жанров
label_encoder = data['label_encoder']  # для обратного декодирования
```

**Что сохраняется в кэш (`data/processed/`):**

| Файл | Формат | Содержание |
|------|--------|------------|
| `X_train.npy` | NumPy | Признаки train |
| `X_val.npy` | NumPy | Признаки val |
| `X_test.npy` | NumPy | Признаки test |
| `y_*.npy` | NumPy | Метки |
| `preprocessor.pkl` | pickle | LabelEncoder + StandardScaler |
| `metadata.json` | JSON | Информация о датасете |
| `pipeline_medium_min100.pkl` | pickle | Полный объект (для быстрой загрузки) |

---

### 5. `load_processed.py` — быстрая загрузка

**Что делает:**
Загружает уже обработанные данные из кэша без повторного запуска пайплайна.

**Как использовать:**

```python
# Самый простой способ
from src.data.load_processed import load_data

data = load_data()  # использует настройки из config (FMA Medium)

X_train = data['X_train']
y_train = data['y_train']

# С параметрами
data = load_data(
    subset="small",              # FMA Small
    min_samples_per_genre=50,    # более мягкая фильтрация
    as_dataframe=False           # вернуть numpy (по умолчанию)
)

# Если нужен DataFrame (для анализа)
data_df = load_data(as_dataframe=True)
X_train_df = data_df['X_train']  # pd.DataFrame с именами признаков
```

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_data()` | Однострочник для загрузки |
| `load()` | Алиас для `load_data()` |
| `LoadProcessedData` | Класс для более гибкого управления |

---

## 🔄 Типичный workflow

### Первый раз (подготовка данных)

```python
from src.data.pipeline import DataPipeline

# Запуск пайплайна (долго, 1-2 минуты)
pipeline = DataPipeline(subset="medium", min_samples_per_genre=100)
data = pipeline.run()

# Данные сохранены в data/processed/
```

### Все последующие разы (быстрая загрузка)

```python
from src.data.load_processed import load_data

# Мгновенная загрузка из кэша
data = load_data()
X_train, y_train = data['X_train'], data['y_train']
```

---

## 📊 Формат выходных данных

После выполнения пайплайна вы получаете словарь:

```python
{
    'X_train': np.ndarray,   # (n_train, n_features) ~ (6400, 518)
    'X_val': np.ndarray,     # (n_val, n_features)   ~ (800, 518)
    'X_test': np.ndarray,    # (n_test, n_features)  ~ (800, 518)
    
    'y_train': np.ndarray,   # (n_train,) закодированные метки (0..n_classes-1)
    'y_val': np.ndarray,
    'y_test': np.ndarray,
    
    'label_encoder': LabelEncoder,  # для преобразования чисел → жанры
    'scaler': StandardScaler,       # для нормализации новых данных
    
    'genre_names': list,            # ['Electronic', 'Rock', ...]
    
    'metadata': {
        'subset': 'medium',
        'num_samples': 8000,
        'num_features': 518,
        'num_classes': 16,
        'class_names': [...],
        'train_size': 6400,
        'val_size': 800,
        'test_size': 800,
        'min_samples_per_genre': 100,
        'use_features': True,
        'split_method': 'official_fma_80_10_10',
        'feature_names': [...]   # имена колонок (если доступны)
    }
}
```

---

## ⚙️ Конфигурация

Все пути и параметры хранятся в YAML файлах в `configs/`:

### `configs/paths.yaml`
```yaml
external_data:
  metadata_dir: "E:/music-genre-classifier/fma_metadata"
  active_subset: "medium"

project_dirs:
  data:
    raw: "./data/raw"
    processed: "./data/processed"
  ...
```

### `configs/audio.yaml`
```yaml
sample_rate: 22050
duration: 30
n_mels: 128
n_mfcc: 20
...
```

---

## 🧪 Тестирование

Для модуля данных написаны тесты, которые запускаются на реальных данных **FMA Small**:

```bash
pytest tests/data/ -v
```

Тесты проверяют:
- Загрузку всех CSV файлов
- Фильтрацию редких жанров
- Кодирование и нормализацию
- Работу пайплайна и кэширование
- Сохранение/загрузку препроцессора

---

## 🚀 Пример использования в ноутбуке

```python
# Быстрая загрузка данных
from src.data.load_processed import load_data

data = load_data()
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
genre_names = data['genre_names']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Жанры: {', '.join(genre_names)}")
```

---

## 📌 Важные замечания

1. **Первый запуск пайплайна может занять 1-2 минуты** — это нормально
2. **Данные кэшируются** — последующие загрузки мгновенны
3. **Используйте FMA Small для быстрых экспериментов**, FMA Medium для финальных результатов
4. **Никогда не изменяйте данные в `data/processed/` вручную** — используйте пайплайн

---

## 🔗 Связанные файлы

- `configs/paths.yaml` — пути к данным
- `configs/audio.yaml` — параметры аудио
- `tests/data/` — тесты модуля
- `notebooks/01_load_and_explore.ipynb` — пример использования
