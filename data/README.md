# ClaimLens Dataset Guide

This directory contains the dataset files used for machine learning model training and evidence-assisted verification.

## 1. Datasets Included in ClaimLens

### `curated_evidence.csv`
A curated collection of 12 documented fact-checks from authoritative organizations (Reuters Fact Check, AP News, Snopes, PolitiFact, WHO, NASA, NATO, ISRO).
- **Format:** CSV with fields `claim_id`, `claim_text`, `topic`, `documented_assessment`, `publisher`, `source_url`, `date_checked`, `supporting_summary`, `relevant_passages`, `irrelevant_passages`.
- **Use:** Used by `factcheck.py` and `relevance.py` for offline fact verification and passage relevance detection.

### `sample_articles.csv`
Three preloaded demonstration scenarios covering:
- **Scenario A (Likely Real):** ISRO Chandrayaan-3 Moon Landing.
- **Scenario B (Likely Fake):** NASA 15 Days of Total Darkness Hoax.
- **Scenario C (Unverified):** Springfield Municipal Council Streetlight Procurement.

### `train_dataset.csv`
A verified initial benchmark dataset of real news and documented misinformation articles compiled from public benchmark datasets (ISOT Fake News Dataset / McIntire / Reuters public wire archives / Fact-check databases).
- **Columns:**
  - `text`: Headline and article body text.
  - `label`: Binary classification target (`1` for Real News, `0` for Fake News).
  - `source`: Publisher / source origin.
  - `topic`: News category (Politics, Science, Technology, Health, General).

## 2. Using Larger External Datasets (e.g., Kaggle / ISOT)

The `train_model.py` script is designed to handle large external datasets seamlessly.

### Option A: Standard Single CSV
If you download a CSV with text and label columns (such as the Kaggle McIntire Fake News Dataset):
```bash
py -3.11 train_model.py --data data/train_dataset.csv --text-col text --label-col label
```

### Option B: Separate True.csv and Fake.csv (ISOT Dataset)
The popular University of Victoria ISOT Fake News Dataset distributes real and fake articles in two separate files (`True.csv` and `Fake.csv`):
1. Download `True.csv` and `Fake.csv` from Kaggle or the ISOT project page.
2. Place both files in the `data/` directory:
   - `data/True.csv`
   - `data/Fake.csv`
3. Run the training command:
```bash
py -3.11 train_model.py --true-csv data/True.csv --fake-csv data/Fake.csv
```

The script will automatically load both files, assign `1` to True and `0` to Fake, deduplicate texts, split into train and test sets, fit the TF-IDF vectorizer, train Logistic Regression, evaluate the model, and save `models/claimlens_model.joblib` and `models/model_metadata.json`.
