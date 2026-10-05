# Breast Cancer Detection using KNN

A machine learning web app that classifies a tumour as **benign** or **malignant** from cell nucleus measurements, using the **K-Nearest Neighbors (KNN)** algorithm and a **Flask** web interface.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis. Always consult a qualified doctor.

## Features

- KNN classifier (K = 5) trained on all 30 features, with `StandardScaler` feature scaling
- Label encoding of the target (B = 0, M = 1) and a stratified train/test split
- Evaluation with accuracy, confusion matrix, precision, recall and F1 score
- Web form with 30 inputs grouped into Mean, Standard error and Worst values, plus buttons to load typical benign and malignant sample values
- Shows the prediction and how many of the K nearest neighbours agree
- Input validation and an on-page medical disclaimer

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** dataset: 569 samples, 30 numeric features computed from images of cell nuclei.

| Item | Detail |
|---|---|
| Target | `diagnosis`: B = Benign (357), M = Malignant (212) |
| Features used | All 30: for each of 10 measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the mean, standard error (`_se`) and worst value |
| Dropped columns | `id` (identifier) and `Unnamed: 32` (empty column) |
| Missing values | None |

## Project Structure

```
breast_cancer_knn/
├── app.py                 # Flask backend: loads the model and serves predictions
├── model.py               # Cleans data, trains KNN, evaluates, saves model.pkl
├── diagnosis.csv          # Dataset
├── model.pkl              # Saved scaler + KNN model (created by model.py)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web page
└── README.md
```

## How It Works

1. `model.py` loads `diagnosis.csv`, drops useless columns and converts `diagnosis` to numbers (M = 1, B = 0).
2. The data is split 80% / 20% into training and test sets (`random_state=42`, stratified so both sets keep the same benign/malignant ratio).
3. `StandardScaler` is fitted on the training data only, then applied to both training and test data.
4. A KNN classifier (K = 5) is trained on the scaled training data and evaluated on the unseen test set.
5. The scaler, the model, the feature names and the class names are saved together in `model.pkl`.
6. `app.py` loads them. When the user submits the form, the values are validated, scaled with the saved scaler and classified by KNN.

```
Browser (index.html) --> Flask (app.py) --> scaler --> KNN --> prediction --> Browser
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/PatelVishakha-07/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates model.pkl)
python model.py

# 4. Start the web app
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Click a sample button or enter your own values, then press **Predict**.

> If you retrain the model, restart the app so it loads the new `model.pkl`.

## Results

Results on the 20% test set (114 samples), using `random_state=42` and K = 5:

| Metric | Value |
|---|---|
| K | 5 |
| Test accuracy | 95.6% |
| Benign precision / recall | 0.95 / 0.99 |
| Malignant precision / recall | 0.97 / 0.90 |

Confusion matrix:

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 71 | 1 |
| **Actual Malignant** | 4 | 38 |

Exact numbers may differ slightly with other library versions. Recall for the malignant class matters most in this setting, since a missed malignant case is the costliest mistake.

## Why Feature Scaling Matters

KNN predicts using distances between samples. Features such as `area_mean` (hundreds to thousands) would dominate features such as `smoothness_mean` (around 0.1) if the data were not scaled. `StandardScaler` puts every feature on the same scale. The scaler learns only from training data, and the same scaler is applied to new inputs in the web app.

## Limitations

- The web form needs 30 values, which is hard for a non-expert to provide; it is meant as a demo (use the sample buttons).
- K = 5 is the default and has not been tuned.
- The dataset is small (569 samples) and imbalanced (about 63% benign).
- Values entered by users must come from the same kind of measurement as the training data.
- KNN stores all training data and compares against it at prediction time, which can be slow for very large datasets.

## Future Improvements

- Compare KNN with Logistic Regression, SVM and Random Forest
- Tune K with cross-validation (`GridSearchCV`)
- Reduce to the most important features for a shorter form
- Add ROC curve and precision-recall charts to the page
- Deploy the app online (Render, PythonAnywhere)

## Common Issues

| Problem | Fix |
|---|---|
| `TemplateNotFound: index.html` | Keep `index.html` inside a folder named `templates`, next to `app.py`. |
| `FileNotFoundError: model.pkl` | Run `python model.py` first. |
| Old predictions after retraining | Restart `app.py`. |

## Author

**Vishakha Pareshkumar Patel**
MCA Student, LJ University, Ahmedabad
[GitHub](https://github.com/PatelVishakha-07) | [LinkedIn](https://www.linkedin.com/in/patelvishakha-tech)
