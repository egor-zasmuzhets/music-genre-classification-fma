# Data Module — Working with FMA Dataset

This module handles loading, preprocessing, and caching data from the [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset.

## 📁 Module Structure

```
src/data/
├── config.py           # Path and parameter configuration
├── loader.py           # Raw FMA CSV file loading
├── preprocessor.py     # Preprocessing: encoding, normalization, filtering
├── pipeline.py         # Complete data preparation pipeline
├── load_processed.py   # Fast loading of already processed data
└── README.md           # This documentation (Russian)
└── README_EN.md        # This documentation (English)
```

---

## 🎯 Main Components

### 1. `config.py` — Configuration Management

**What it does:**
- Finds the project root (looks for `configs/paths.yaml`)
- Loads YAML configuration files
- Provides global objects `paths` and `audio_params`

**How to use:**

```python
from src.data.config import paths, audio_params

# Paths to directories
print(paths.processed_data_dir)   # ./data/processed
print(paths.models_dir)           # ./models

# Audio parameters
print(audio_params.sample_rate)   # 22050
print(audio_params.n_mels)        # 128
```

**Key `paths` properties:**

| Property | Purpose |
|----------|---------|
| `metadata_dir` | Directory with `tracks.csv`, `features.csv` |
| `raw_data_dir` | For raw audio files (MP3) |
| `processed_data_dir` | For cached data (`.npy`, `.pkl`) |
| `checkpoints_dir` | For model weights |
| `results_dir` | For plots and metrics |

---

### 2. `loader.py` — Loading FMA Metadata

**What it does:**
- Loads `tracks.csv` (multi-index, 2 levels)
- Loads `features.csv` (518 precomputed features)
- Loads `genres.csv` (genre hierarchy)
- Filters tracks by subset (small/medium/large)
- Extracts official train/val/test split

**How to use:**

```python
from src.data.loader import FMALoader

loader = FMALoader()

# Load all FMA Medium tracks
tracks = loader.get_tracks_by_subset("medium")

# Get features
features = loader.features  # (106574, 518)

# Get split
splits = loader.get_available_splits(tracks)
train_idx = splits['training']   # track indices for training
```

**Features:**
- `tracks.csv` has multi-index: columns like `('track', 'genre_top')`
- `features.csv` has triple multi-index: `('feature', 'method', 'statistic')`
- Loading is **lazy** (data loads on first access)

---

### 3. `preprocessor.py` — Data Preprocessing

**What it does:**
- Filters rare genres (by `min_samples_per_genre`)
- Encodes text labels to numbers (`LabelEncoder`)
- Normalizes features (`StandardScaler`)
- Computes class weights for imbalance handling

**How to use:**

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(min_samples_per_genre=100)

# 1. Filter rare genres
tracks_filtered = preprocessor.filter_rare_genres(tracks, ('track', 'genre_top'))

# 2. Encode labels (fit on train, transform on val/test)
y_train_enc, y_val_enc, y_test_enc = preprocessor.encode_labels(
    y_train, y_val, y_test
)

# 3. Normalize features
X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
    X_train, X_val, X_test
)

# 4. Class weights
class_weights = preprocessor.get_class_weights(y_train_enc)
```

**Important details:**
- `fit` is performed **only on training data**
- `transform` is applied to validation and test sets (no data leakage)
- Class weights use `balanced` formula: `weight = n_samples / (n_classes * n_class_samples)`

---

### 4. `pipeline.py` — Complete Pipeline

**What it does:**
Combines all steps into one automated process:

1. Load FMA Small/Medium/Large tracks
2. Filter tracks without genre labels
3. Remove rare genres
4. Use official FMA split (80/10/10)
5. Match features from `features.csv` with tracks
6. Encode labels and normalize features
7. Save to cache (`.npy` + `.pkl`)

**How to use:**

```python
from src.data.pipeline import DataPipeline

# Create pipeline for FMA Medium
pipeline = DataPipeline(
    subset="medium",              # small, medium, large
    min_samples_per_genre=100,    # minimum tracks per genre
    use_features=True,            # use precomputed features
)

# Run (first run is slow, then loads from cache)
data = pipeline.run()

# Extract data
X_train = data['X_train']   # numpy array, normalized features
X_val   = data['X_val']
X_test  = data['X_test']
y_train = data['y_train']   # encoded labels (0..num_classes-1)
y_val   = data['y_val']
y_test  = data['y_test']

# Additional info
genre_names = data['genre_names']   # list of genre names
label_encoder = data['label_encoder']  # for reverse decoding
```

**Cache files (`data/processed/`):**

| File | Format | Content |
|------|--------|---------|
| `X_train.npy` | NumPy | Train features |
| `X_val.npy` | NumPy | Validation features |
| `X_test.npy` | NumPy | Test features |
| `y_*.npy` | NumPy | Labels |
| `preprocessor.pkl` | pickle | LabelEncoder + StandardScaler |
| `metadata.json` | JSON | Dataset information |
| `pipeline_medium_min100.pkl` | pickle | Full object (fast loading) |

---

### 5. `load_processed.py` — Fast Loading

**What it does:**
Loads already processed data from cache without re-running the pipeline.

**How to use:**

```python
# Simplest way
from src.data.load_processed import load_data

data = load_data()  # uses config settings (FMA Medium)

X_train = data['X_train']
y_train = data['y_train']

# With parameters
data = load_data(
    subset="small",              # FMA Small
    min_samples_per_genre=50,    # softer filtering
    as_dataframe=False           # return numpy (default)
)

# If DataFrame is needed (for analysis)
data_df = load_data(as_dataframe=True)
X_train_df = data_df['X_train']  # pd.DataFrame with feature names
```

**Functions:**

| Function | Description |
|----------|-------------|
| `load_data()` | One-liner for loading |
| `load()` | Alias for `load_data()` |
| `LoadProcessedData` | Class for more flexible control |

---

## 🔄 Typical Workflow

### First time (data preparation)

```python
from src.data.pipeline import DataPipeline

# Run pipeline (slow, 1-2 minutes)
pipeline = DataPipeline(subset="medium", min_samples_per_genre=100)
data = pipeline.run()

# Data saved to data/processed/
```

### All subsequent runs (fast loading)

```python
from src.data.load_processed import load_data

# Instant loading from cache
data = load_data()
X_train, y_train = data['X_train'], data['y_train']
```

---

## 📊 Output Data Format

After running the pipeline, you get a dictionary:

```python
{
    'X_train': np.ndarray,   # (n_train, n_features) ~ (6400, 518)
    'X_val': np.ndarray,     # (n_val, n_features)   ~ (800, 518)
    'X_test': np.ndarray,    # (n_test, n_features)  ~ (800, 518)
    
    'y_train': np.ndarray,   # (n_train,) encoded labels (0..n_classes-1)
    'y_val': np.ndarray,
    'y_test': np.ndarray,
    
    'label_encoder': LabelEncoder,  # for converting numbers → genres
    'scaler': StandardScaler,       # for normalizing new data
    
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
        'feature_names': [...]   # column names (if available)
    }
}
```

---

## ⚙️ Configuration

All paths and parameters are stored in YAML files in `configs/`:

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

## 🧪 Testing

Tests for the data module run on real **FMA Small** data:

```bash
pytest tests/data/ -v
```

Tests verify:
- Loading all CSV files
- Filtering rare genres
- Encoding and normalization
- Pipeline and caching behavior
- Preprocessor save/load functionality

---

## 🚀 Usage Example in Notebook

```python
# Fast data loading
from src.data.load_processed import load_data

data = load_data()
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
genre_names = data['genre_names']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Genres: {', '.join(genre_names)}")
```

---

## 📌 Important Notes

1. **First pipeline run may take 1-2 minutes** — this is normal
2. **Data is cached** — subsequent loads are instant
3. **Use FMA Small for quick experiments**, FMA Medium for final results
4. **Never manually modify data in `data/processed/`** — use the pipeline

---

## 🔗 Related Files

- `configs/paths.yaml` — data paths
- `configs/audio.yaml` — audio parameters
- `tests/data/` — module tests
- `notebooks/01_load_and_explore.ipynb` — usage example