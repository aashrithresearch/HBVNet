
# HBV vs Non-Viral HCC Classifier (Minimal Signature)

## 📌 Overview
This project classifies **HBV-induced hepatocellular carcinoma (HCC)** vs **non-viral HCC** using clinical and transcriptomic data from **TCGA-LIHC**.

We implemented a machine-learning pipeline including preprocessing, feature selection, modeling, and evaluation, then deployed the model as a **Gradio app**.

## 📊 Dataset
- **Source:** [TCGA-LIHC via UCSC Xena](https://xenabrowser.net/datapages/?cohort=TCGA%20Liver%20Cancer%20(LIHC))
- **Samples:** ~325 patients with merged clinical + RNA-seq (FPKM-UQ)
- **Labels:** HBV-positive vs Non-viral

## 🛠 Workflow
1. **Preprocessing:** Harmonized TCGA barcodes, merged clinical + expression data.
2. **Encoding:** Gender binary-encoded (0/1), stage and grade ordinal-encoded, missing values imputed.
3. **Feature Selection:** Top 500 variable genes → LASSO logistic regression → ~15-gene minimal signature.
4. **Balancing:** Applied SMOTE to handle class imbalance.
5. **Modeling:** XGBoost trained on Clinical-only, Expression-only, Combined, Minimal Signature.
6. **Evaluation:** ROC-AUC, F1, Accuracy + ROC curve, confusion matrix, calibration curve.
7. **Stretch Goals:** Modality comparison, minimal signature retention, Gradio app deployment.

## 📈 Performance (Example)
| Modality            | ROC-AUC | F1   | Accuracy |
|---------------------|---------|------|----------|
| Clinical-only       | 0.81    | 0.74 | 0.78     |
| Expression-only     | 0.83    | 0.76 | 0.80     |
| Clinical+Expr       | 0.87    | 0.79 | 0.83     |
| Minimal Signature   | 0.86    | 0.78 | 0.82     |

## 📊 Figures
Saved plots:
- ROC curve (`minimal_signature_roc.png`)
- Confusion matrix (`minimal_signature_cm.png`)
- Feature importance plot

## 🚀 Usage
### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Model + Regenerate Report
Open the notebook and run all cells to preprocess data, train model, and save artifacts.  
Generated report PDF will be saved in `/mnt/data/`.

### 3. Launch Gradio App
```bash
python app.py
```
Open the local URL or public share link to use the interactive classifier.

## 🔄 Retraining & Regenerating Artifacts
To retrain the minimal signature model and regenerate the files used by the Gradio app:

```python
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Use selected_genes from LASSO
X_sig = X_expr[selected_genes]
X_train, X_test, y_train, y_test = train_test_split(
    X_sig, y, stratify=y, test_size=0.3, random_state=42
)

# Train XGBoost
xgb_sig = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, eval_metric="auc"
)
xgb_sig.fit(X_train, y_train)

# Save artifacts
joblib.dump(xgb_sig, "xgb_minimal_signature.pkl")
joblib.dump(list(X_train.columns), "minimal_signature_genes.pkl")
joblib.dump(X_train.mean(), "minimal_signature_means.pkl")
```

This ensures your **model**, **gene list**, and **means** are in sync with your dataset.

## 🧠 Interpretation
- Stage and Grade are strong predictors of HBV etiology.
- Immune-related and proliferation genes are enriched in HBV cases.
- Combined modality outperforms single modalities, showing added value of expression data.

---
Hackathon Project © 2025 | Team XYZ


## 🖥 Using the Gradio App
The repository includes an `app.py` file to run the interactive predictor.

### Run Locally
```bash
python app.py
```
You will see a local URL (e.g., http://127.0.0.1:7860). Open it in a browser to adjust gene values and view predicted probability and classification.
