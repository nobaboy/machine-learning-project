## Dependencies

This project uses standard machine learning and data science libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn`, alongside a few extra dependencies including `pyarrow` and `seaborn`.

All required packages are listed in `requirements.txt` and can be installed by running:

```
pip install -r requirements.txt
```

## Running

The project is run from the `__main__.py` script, the project does not auto generate a run configuration for IDEs such as PyCharm or VS Code.

Creating a run configuration or running `__main__.py` from the terminal should work fine (e.g. `python __main__.py`)

## Workload Distribution

This project was developed by two contributers:
 - nobaboy (Abdulrahman) ~55%
 - imAboody (Abdullah) ~45%

This list is not exhaustive as we both worked all around the project, fixing many random bugs, and completing missing parts of each others implementations.

### Abdulrahman
 - Dataset loading and optimization `utils/loader.py`
 - Some visualization `utils/visualization.py#L19-L58`
 - Some missing parts of feature engineering
 - Implementation of Task A (Classification)
 - Project cleanup, organization and structure

Estimated understanding ~65%

### Abdullah
 - Some visualization `utils/visualization.py#L62-L162`
 - Data preprocessing pipeline `utils/preprocessing.py`
 - The majority of feature engineering `utils/feature.py`
 - Implementation of Task B (Regression)

Estimated understanding ~70%

Both contributors should cover the entirety of the project's understanding.
