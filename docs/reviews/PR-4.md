# Code Review Report - PR #4

> **PR:** CNN on MFCC data
> **Author:** @egor-zasmuzhets
> **Generated:** 2026-06-11 08:28:18 UTC

## Summary

| Severity | Count |
|----------|-------|
| 🔴 High | 0 |
| 🟡 Medium | 7 |
| 🟢 Low | 18 |
| **Total** | 25 |

## Detailed Issues


### 🟢 STYLE (line 16)

| Property | Value |
|----------|-------|
| **File** | `configs/audio.yaml` |
| **Code** | `augmentation:` |
| **Description** | The indentation is inconsistent, it's better to use a consistent number of spaces for indentation throughout the file. |
| **Suggestion** | Use 2 or 4 spaces for indentation consistently. |


### 🟢 STYLE (line 1)

| Property | Value |
|----------|-------|
| **File** | `configs/audio.yaml` |
| **Code** | `sample_rate: 22050` |
| **Description** | It's a good practice to include a header or a comment at the top of the file to describe the purpose of the configuration. |
| **Suggestion** | Add a comment at the top of the file to describe the purpose of the configuration. |


### 🟢 STYLE (line 10)

| Property | Value |
|----------|-------|
| **File** | `configs/models.yaml` |
| **Code** | `checkpoints_dir: #None` |
| **Description** | The checkpoints_dir is commented out, which might be a typo or a placeholder. It should be either a valid directory path or explicitly set to null. |
| **Suggestion** | Set checkpoints_dir to a valid directory path or remove the comment to set it to null. |


### 🟡 PERFORMANCE (line 24)

| Property | Value |
|----------|-------|
| **File** | `configs/models.yaml` |
| **Code** | `n_jobs: -1` |
| **Description** | Using n_jobs=-1 can lead to high memory usage and slow down the system. It's better to set it to a reasonable value based on the available CPU cores. |
| **Suggestion** | Set n_jobs to a reasonable value, such as the number of available CPU cores minus one. |


### 🟢 STYLE (line 27)

| Property | Value |
|----------|-------|
| **File** | `configs/models.yaml` |
| **Code** | `use_class_weights: true` |
| **Description** | The value 'true' should be enclosed in quotes as it's a string in YAML. |
| **Suggestion** | Enclose 'true' in quotes: "true". |


### 🟢 STYLE (line 1)

| Property | Value |
|----------|-------|
| **File** | `configs/paths.yaml` |
| **Code** | `project_root: "/home/yahor/PycharmProjects/PythonProject/music-genre-classification-fma"` |
| **Description** | The project root directory path is hardcoded and may not be portable across different environments. |
| **Suggestion** | Consider using environment variables or relative paths to make the project more portable. |


### 🟢 STYLE (line 9)

| Property | Value |
|----------|-------|
| **File** | `configs/paths.yaml` |
| **Code** | `active_subset: "medium"` |
| **Description** | The active subset is hardcoded and may need to be changed frequently. |
| **Suggestion** | Consider using a command-line argument or an environment variable to make the active subset more configurable. |


### 🟢 SECURITY (line 10)

| Property | Value |
|----------|-------|
| **File** | `configs/paths.yaml` |
| **Code** | `active_zip: "/home/yahor/PycharmProjects/PythonProject/music-genre-classification-fma/data/raw/fma_m` |
| **Description** | The active zip file path is hardcoded and may pose a security risk if the file is not properly validated. |
| **Suggestion** | Consider using a secure method to validate the zip file and its contents before using it. |


### 🟢 STYLE (line 1)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/adjusted_analysis/adjusted_results.json` |
| **Code** | `{` |
| **Description** | The JSON object is not formatted with indentation, making it difficult to read. |
| **Suggestion** | Use a JSON formatter to indent the object for better readability. |


### 🟡 PERFORMANCE (line 153)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/adjusted_analysis/adjusted_results.json` |
| **Code** | `"adjusted_confusion_matrix": [` |
| **Description** | The confusion matrix is very large and may cause performance issues when parsing or processing the JSON object. |
| **Suggestion** | Consider using a more compact representation of the confusion matrix or splitting it into separate objects. |


### 🟢 STYLE (line 7)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/adjusted_analysis/adjusted_results.json` |
| **Code** | `"corrections_applied": 13,` |
| **Description** | The property name 'corrections_applied' is not descriptive and may not be clear to other developers. |
| **Suggestion** | Consider renaming the property to something more descriptive, such as 'num_corrections_applied'. |


### 🟢 STYLE (line 1)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/adjusted_analysis/final_adjusted_results.json` |
| **Code** | `{` |
| **Description** | The JSON object is not formatted with indentation, making it difficult to read. |
| **Suggestion** | Use a JSON formatter to indent the object for better readability. |


### 🟢 PERFORMANCE (line 157)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/adjusted_analysis/final_adjusted_results.json` |
| **Code** | `"confusion_matrix": [` |
| **Description** | The confusion matrix is a large array, which could impact performance if the data is very large. |
| **Suggestion** | Consider using a more efficient data structure, such as a sparse matrix, if the data is very large. |


### 🟡 BUG (line 446)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/adjusted_analysis/final_adjusted_results.json` |
| **Code** | `}` |
| **Description** | The JSON object does not contain any error handling or validation, which could lead to errors if the data is malformed. |
| **Suggestion** | Add error handling and validation to ensure the data is correct and handle any errors that may occur. |


### 🟢 STYLE (line 1)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/real_analysis/results.json` |
| **Code** | `{` |
| **Description** | The JSON object is not formatted with indentation, making it difficult to read. |
| **Suggestion** | Use a consistent indentation scheme to improve readability. |


### 🟢 STYLE (line 8)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/real_analysis/results.json` |
| **Code** | `"per_class_f1": {` |
| **Description** | The 'per_class_f1' object has a large number of properties, making it difficult to read. |
| **Suggestion** | Consider breaking this object into smaller, more manageable pieces. |


### 🟢 PERFORMANCE (line 25)

| Property | Value |
|----------|-------|
| **File** | `presentation_figures/real_analysis/results.json` |
| **Code** | `}` |
| **Description** | The JSON object contains a large number of decimal places, which may impact performance when parsing or serializing the data. |
| **Suggestion** | Consider rounding the decimal values to a reasonable number of places. |


### 🟢 STYLE (line 9)

| Property | Value |
|----------|-------|
| **File** | `src/data/__init__.py` |
| **Code** | `if TYPE_CHECKING:` |
| **Description** | The use of TYPE_CHECKING is not necessary in this context, as it is only used for static type checking and does not affect the runtime behavior of the code. |
| **Suggestion** | Consider removing the TYPE_CHECKING block for simplicity. |


### 🟡 PERFORMANCE (line 70)

| Property | Value |
|----------|-------|
| **File** | `src/data/__init__.py` |
| **Code** | `module = __import__(module_name, fromlist=[name])` |
| **Description** | The use of __import__ can lead to performance issues and potential security vulnerabilities if not used carefully. |
| **Suggestion** | Consider using importlib instead of __import__ for more control and flexibility. |


### 🟢 STYLE (line 78)

| Property | Value |
|----------|-------|
| **File** | `src/data/__init__.py` |
| **Code** | `logger = logging.getLogger(__name__)` |
| **Description** | The logger is not being used anywhere in the code, making it unnecessary. |
| **Suggestion** | Consider removing the unused logger. |


### 🟡 PERFORMANCE (line 89)

| Property | Value |
|----------|-------|
| **File** | `src/data/loader.py` |
| **Code** | `low_memory=False` |
| **Description** | The use of low_memory=False in pd.read_csv can lead to memory issues for large datasets. |
| **Suggestion** | Consider using chunksize to process the data in chunks. |


### 🟢 STYLE (line 50)

| Property | Value |
|----------|-------|
| **File** | `src/data/loader.py` |
| **Code** | `def __init__(self, metadata_dir: Optional[Path] = None) -> None:` |
| **Description** | The __init__ method does not follow the standard Python docstring convention. |
| **Suggestion** | Consider adding a docstring to the __init__ method. |


### 🟡 BUG (line 203)

| Property | Value |
|----------|-------|
| **File** | `src/data/loader.py` |
| **Code** | `raise ValueError(f"Unknown subset: '{subset}'. ")` |
| **Description** | The error message does not provide enough information about the valid subset values. |
| **Suggestion** | Consider adding a list of valid subset values to the error message. |


### 🟢 STYLE (line 278)

| Property | Value |
|----------|-------|
| **File** | `src/data/loader.py` |
| **Code** | `def print_info(self) -> None:` |
| **Description** | The print_info method does not follow the standard Python docstring convention. |
| **Suggestion** | Consider adding a docstring to the print_info method. |


### 🟡 SECURITY (line 83)

| Property | Value |
|----------|-------|
| **File** | `src/data/loader.py` |
| **Code** | `raise FileNotFoundError(f"tracks.csv not found: {tracks_path}")` |
| **Description** | The error message reveals the path to the tracks.csv file, which could be a security risk. |
| **Suggestion** | Consider removing the file path from the error message. |

