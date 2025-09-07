import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load artifacts
MODEL_PATH = "xgb_minimal_signature.pkl"
GENES_PATH = "minimal_signature_genes.pkl"
MEANS_PATH = "minimal_signature_means.pkl"

xgb_model = joblib.load(MODEL_PATH)
minimal_genes = list(joblib.load(GENES_PATH))

means_series = joblib.load(MEANS_PATH)
if not isinstance(means_series, pd.Series):
    means_series = pd.Series(means_series, index=minimal_genes, dtype=float)
means_series = means_series.reindex(minimal_genes).fillna(1.0)

try:
    model_feature_names = xgb_model.get_booster().feature_names
    if model_feature_names is None:
        model_feature_names = minimal_genes
except Exception:
    model_feature_names = minimal_genes

def predict_from_list(*vals):
    try:
        vals = [np.nan if v is None else float(v) for v in vals]
        input_df = pd.DataFrame([dict(zip(minimal_genes, vals))])
        input_df = input_df.reindex(columns=model_feature_names)
        input_df = input_df.fillna(means_series.reindex(model_feature_names))
        input_df = input_df.astype(float)
        prob = float(xgb_model.predict_proba(input_df)[:, 1][0])
        label = "HBV-induced HCC" if prob >= 0.5 else "Non-viral HCC"
        return prob, label
    except Exception as e:
        return 0.0, f"Error: {str(e)}"

inputs = [gr.Number(label=gene, value=float(means_series.get(gene, 1.0)), precision=4)
          for gene in minimal_genes]

iface = gr.Interface(
    fn=predict_from_list,
    inputs=inputs,
    outputs=[gr.Number(label="Predicted Probability"), gr.Textbox(label="Prediction")],
    title="HBV vs Non-Viral HCC Classifier (Minimal Gene Signature)",
    description="Provide expression values for the minimal gene signature. Pre-filled with cohort means for convenience."
)

if __name__ == "__main__":
    iface.launch(share=True)
