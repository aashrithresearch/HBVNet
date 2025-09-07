
# HBV vs Non-Viral HCC Classifier (Minimal Signature)

## Overview
This project classifies **HBV-induced hepatocellular carcinoma (HCC)** vs **non-viral HCC** using clinical and transcriptomic data from **TCGA-LIHC**.

We implemented a machine-learning pipeline including preprocessing, feature selection, modeling, and evaluation, then deployed the model as a **Gradio app**.

## Dataset
- **Source:** [TCGA-LIHC via UCSC Xena](https://xenabrowser.net/datapages/?cohort=TCGA%20Liver%20Cancer%20(LIHC))
- **Samples:** ~325 patients with merged clinical + RNA-seq (FPKM-UQ)
- **Labels:** HBV-positive vs Non-viral

## Performance 
| Modality            | ROC-AUC | F1   | Accuracy |
|---------------------|---------|------|----------|
| Clinical-only       | 0.68    | 0.83 | 0.72     |
| Expression-only     | 0.68    | 0.70 | 0.82     |
| Clinical+Expr       | 0.69    | 0.71 | 0.82     |
| Minimal Signature   | 0.71    | 0.77 | 0.85     |

## Usage
### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Model + Regenerate Report
Open the notebook and run all cells to preprocess data, train model, and save artifacts. Download clinical metadata, and attach via files.upload 

### 3. Launch Gradio App
Open the local URL linked in the ``app.ipynb`` or public share link to use the interactive classifier.

## Retraining & Regenerating Artifacts
To retrain the minimal signature model and regenerate the files used by the Gradio app:

```python
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X_sig = X_expr[selected_genes]
X_train, X_test, y_train, y_test = train_test_split(
    X_sig, y, stratify=y, test_size=0.3, random_state=42
)

xgb_sig = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, eval_metric="auc"
)
xgb_sig.fit(X_train, y_train)

joblib.dump(xgb_sig, "xgb_minimal_signature.pkl")
joblib.dump(list(X_train.columns), "minimal_signature_genes.pkl")
joblib.dump(X_train.mean(), "minimal_signature_means.pkl")
```

This ensures **model**, **gene list**, and **means** are in sync with your dataset.
