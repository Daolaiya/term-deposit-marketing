# Term Deposit Marketing
## Introduction
**Background:**  
- ACME is a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.
- We are interested in developing a robust machine learning system that leverages information coming from call center data.
- Ultimately, at ACME we are looking to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

**Data Description:**  
- The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

**Attributes:**
- age: age of customer (numeric)
- job: type of job (categorical)
- marital: marital status (categorical)
- education (categorical)
- default: has credit in default? (binary)
- balance: average yearly balance, in euros (numeric)
- housing: has a housing loan? (binary)
- loan: has personal loan? (binary)
- contact: contact communication type (categorical)
- day: last contact day of the month (numeric)
- month: last contact month of year (categorical)
- duration: last contact duration, in seconds (numeric)
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- Output (desired target):
    - y: has the client subscribed to a term deposit? (binary)

**Download Data:**
- https://drive.google.com/file/d/1EW-XMnGfxn-qzGtGPa3v_C63Yqj2aGf7

**Goal(s):**
- Predict if the customer will subscribe (yes/no) to a term deposit (variable y).

**Success Metric(s)**
- Hit 81% or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.

**Bonus(es):**
- We are also interested in finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
- What makes the customers buy? Tell us which feature we should be focusing more on.

## Project Structure
```
TDM/
├── notebook_tdm_1.ipynb   # EDA, preprocessing, model training & saving
├── notebook_tdm_2.ipynb   # Feature importance, customer segmentation, clustering
├── utils.py               # Shared helper functions imported by both notebooks
├── data_raw.csv           # Source dataset (40,000 rows × 14 columns)
├── models/                # Pickled models trained on the full feature set
│   ├── LR.pkl    GNB.pkl    DTC.pkl    RF.pkl
│   └── XGB.pkl   CAT.pkl    LGBM.pkl
├── models_dropped/        # Pickled models trained on the reduced feature set
│   └── (same filenames as above)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
This project targets Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## How to Run
1. Make sure `data_raw.csv` is at the project root (download link above).
2. Launch Jupyter and run the notebooks in order:
   ```bash
   jupyter notebook
   ```
3. Run `notebook_tdm_1.ipynb` first — it performs EDA / preprocessing and writes the trained models to `models/` and `models_dropped/`.
4. Then run `notebook_tdm_2.ipynb`, which loads the saved models from `models/` for feature importance and customer segmentation analysis.

## Methodology Notes
- **Train/test split is stratified and performed *before* class balancing.** Only the training fold is under‑sampled to a 50/50 class ratio; the held‑out test fold preserves the natural distribution (~92.7 % `no` / 7.3 % `yes`). This avoids inflating accuracy by evaluating on an artificially balanced test set.
- **5‑fold cross‑validation** is computed on the (balanced) training fold, which is the success metric called out in the goals above.
- **ROC / AUC** are computed on the held‑out test fold using `predict_proba` scores.
- Imputation, label encoding and helper utilities live in `utils.py` and are imported by both notebooks.

## Results Summary
- `CatBoostClassifier`, `LGBMClassifier`, `RandomForestClassifier`, `XGBClassifier` and `DecisionTreeClassifier` cleared the 81 % accuracy target under the original pipeline. **The notebooks have been refactored to fix data leakage and evaluation issues, so the cached output cells are stale until the notebooks are re‑run** — expect different (but more honest) accuracy numbers afterwards.
- Feature‑importance analysis (model attributes, permutation importance, RFE) consistently flags `duration` as the dominant predictor, followed by `month`, `day`, `housing`, `balance` and `age`.
- PCA / t‑SNE projections show no clean linear separability between subscribers and non‑subscribers.