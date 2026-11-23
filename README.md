# Fraud Detection with PyOD AutoEncoder

This assignment uses an unsupervised neural-network AutoEncoder to flag unusual credit card transactions. The extra **amount-group audit** checks whether fraud recall differs for small and large purchases.

## Dataset
Download the **Credit Card Fraud Detection** dataset from https://www.kaggle.com/datasets/whenamancodes/fraud-detection and place `creditcard.csv` in this folder. The dataset contains anonymized V1–V28 features, Time, Amount, and Class (0 = normal, 1 = fraud). Do not commit the large dataset.

## Run (Python 3.10–3.12 recommended)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python fraud_detection.py --data creditcard.csv --epochs 20
```

If training is slow, try `--epochs 10 --max-train 10000` and document the changed settings in your report.

## Outputs
The console prints evaluation metrics and the amount-group audit. `outputs/metrics.json`, `outputs/test_predictions.csv`, and `outputs/score_distribution.png` are created automatically. Take a screenshot of the real console results and paste it into the Word report. Replace the report's placeholders with your actual values.

## Experiment design
- Stratified 80/20 development/test split; labels are only used to preserve test class balance and for final evaluation.
- AutoEncoder training uses **only normal transactions** from development data; fraud labels are not used for model fitting.
- StandardScaler is fitted only on normal training data; Amount is log-transformed.
- A separate normal-only calibration sample sets a 99.5th-percentile threshold.
- Held-out test data are used once to report precision, recall, average precision, ROC AUC and confusion counts.
- Extra audit checks fraud recall in amount groups: under $50, $50–$499.99, and $500 or more.

**Limitation:** Training on known normal transactions is a semi-supervised anomaly-detection setup (normal-only training), not a completely label-free training-data selection process. The AutoEncoder itself does not train on fraud labels. This is a classroom demonstration, not a production fraud system.

## GitHub steps
1. Create a public repository named `pyod-autoencoder-fraud-audit`.
2. Open a terminal inside this folder.
3. Run:
```bash
git init
git add fraud_detection.py requirements.txt manifest.txt README.md .gitignore Fraud_Detection_Assignment_4_Report.docx
git commit -m "Build PyOD AutoEncoder fraud detection and amount audit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pyod-autoencoder-fraud-audit.git
git push -u origin main
```
4. Replace `YOUR_USERNAME` with your real GitHub username. Copy the repository URL into the report and LMS submission.

## Academic honesty
The report contains placeholders because actual results require executing the code on the real Kaggle dataset. Do not claim the model was run until you have tested it. Cite PyOD, Kaggle, and scikit-learn in your report.


## Easier option: Google Colab
Open `Run_Experiment_in_Colab.ipynb` in Google Colab, run the first two cells, upload `fraud_detection.py` in Colab Files, and run the remaining cells. This downloads the Kaggle dataset and creates real experiment results. The notebook has NOT been executed as part of this package.
