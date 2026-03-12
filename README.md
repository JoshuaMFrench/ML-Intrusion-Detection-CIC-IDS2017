# ML Intrusion Detection System — CIC-IDS2017

A supervised machine learning pipeline for network intrusion detection, trained and evaluated on the CIC-IDS2017 dataset. The project compares Logistic Regression against an untuned and tuned Random Forest classifier, achieving near perfect detection performance on real world network traffic data.

---

## Overview

Traditional Intrusion Detection Systems rely on static signatures that fail against novel or zero day attacks. This project addresses that limitation by training machine learning models to classify network flows as either **benign** or **malicious** based on flow level features without relying on predefined attack signatures.

The pipeline handles the full workflow: data loading, cleaning, label encoding, feature selection, model training, evaluation, and output of all plots and results.

---

## Dataset

This project uses the **CIC-IDS2017** dataset published by the Canadian Institute for Cybersecurity. It contains labeled network flow data covering both normal traffic and the following attack categories:

- DDoS
- Port Scans
- Web Attacks (Brute Force, XSS, SQL Injection)
- Infiltration
- Botnet Activity

Due to the dataset's size (2.8M+ flows), the pipeline samples down to **200,000 instances** to ensure computational efficiency while maintaining statistical significance.

>  The dataset is not included in this repository. Download it from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and place the CSV files in the `/data` folder.

---

## Models

| Model                    | Accuracy | Precision | Recall | F1-Score | AUC    |
|--------------------------|----------|-----------|--------|----------|--------|
| Logistic Regression      | 0.8987   | 0.8521    | 0.9164 | 0.8750   | 0.9753 |
| Random Forest (untuned)  | 0.9976   | 0.9969    | 0.9966 | 0.9967   | 0.9997 |
| Random Forest (tuned)    | 0.9986   | 0.9976    | 0.9986 | 0.9981   | 0.9999 |

>  The tuned Random Forest uses `RandomizedSearchCV` to select the best combination of hyperparameters from a defined parameter grid. Because 20 random combinations are sampled each run, the selected hyperparameters and final metrics may vary slightly between runs. The values in the table above reflect the best results achieved during development.

---

## Pipeline Steps

1. Load and concatenate all CSV files from `/data`
2. Strip whitespace from column headers
3. Remove rows with NaN or infinite values
4. Drop constant/zero variance columns
5. Encode labels — `BENIGN → 0`, any attack `→ 1`
6. Select numeric features only
7. Stratified 70/30 train-test split
8. Train and evaluate all three models
9. Save confusion matrix and ROC curve for each model
10. Save the best model to `/results/best_ids_model.pkl`

---

## Project Structure

```
ML-Intrusion-Detection-CIC-IDS2017/
├── ids_pipeline.py           # Main pipeline script
├── requirements.txt
├── data/                     # Place CIC-IDS2017 CSV files here
│   └── README.md             # Dataset download instructions
├── results/                  # Auto-generated on first run
│   ├── Logistic_Regression_confusion_matrix.png
│   ├── Logistic_Regression_roc_curve.png
│   ├── RandomForest_Untuned_confusion_matrix.png
│   ├── RandomForest_Untuned_roc_curve.png
│   ├── RandomForest_Tuned_confusion_matrix.png
│   ├── RandomForest_Tuned_roc_curve.png
│   └── best_ids_model.pkl
└── report/
    └── IDS_Final_Report.pdf
```

---

## Prerequisites

- Python 3.8+

Install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
pandas
matplotlib
scikit-learn
joblib
```

---

## Running the Pipeline

1. Download the CIC-IDS2017 dataset and place all CSV files in the `/data` folder
2. Run the pipeline:

```bash
python ids_pipeline.py
```

All confusion matrices, ROC curves, and the best saved model will be written to `/results` automatically.

**Estimated runtime:** 2+ hours depending on hardware. The pipeline uses all available CPU cores.

---

## Results

All output plots are saved to `/results` after each run. Each model produces:

- **Confusion Matrix** — visualizes true positives, true negatives, false positives, and false negatives
- **ROC Curve** — plots true positive rate vs false positive rate with AUC score

The tuned Random Forest achieves an AUC of **0.9999**, reducing both false positives and false negatives compared to the untuned version. Logistic Regression, while significantly faster to train, struggled with recall on benign traffic, misclassifying over 5,000 instances.

> Original project completed for CYSE 499  George Mason University

---

## License

This project is licensed under the terms described in the [LICENSE](LICENSE) file.
