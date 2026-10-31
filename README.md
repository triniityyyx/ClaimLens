# ClaimLens — Machine Learning-Based Fake News Detection and Fact Verification System

> **Tagline:** Think Before You Trust.  
> **Secondary Tagline:** Don't just read it. Question it.  
> **Academic Context:** Second-Year B.Tech Computer Science Python Mini Project.  
> **Target Demo Date:** Tuesday, October 27, 2026.  
> **Visual Identity:** Ink + Acid Lime (Investigative Journalism × Contemporary Tech).

---

## 1. Executive Overview

Misinformation spreads rapidly through social networks, messaging applications, and unverified blogs. Most existing automated solutions act as superficial text classifiers: they inspect vocabulary and assign a "True" or "False" label without citing a single primary source.

**ClaimLens** solves this fundamental flaw by decoupling **linguistic pattern analysis** from **empirical evidence verification**:
1. **Linguistic Machine Learning:** A Scikit-learn Pipeline (TF-IDF Vectorizer + Logistic Regression) evaluates stylistic markers, sensationalist hyperbole, and clickbait patterns.
2. **Evidence-Assisted Verification:** Integrates with the Google Fact Check Tools API and an offline curated local fact-check repository containing documented investigations from Reuters, AP News, Snopes, PolitiFact, and WHO.
3. **Evidence Relevance Engine:** Uses TF-IDF and Cosine Similarity to classify retrieved passages into **Relevant**, **Possibly Relevant**, and **Not Relevant**, proving *topical relatedness without confusing relevance with truth*.
4. **Audit History & Analytics:** Powered by SQLite with parameterized queries, full-text search, and live metric tracking.

---

## 2. System Architecture

```
                          [ User Claim / Article Input ]
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
     [ Text Normalization ]                          [ Evidence Retrieval ]
   - Unicode quote cleaning                       - Google Fact Check Tools API
   - Punctuation & space hygiene                  - Curated Local Evidence DB
   - Input length validation                                    │
                │                                               ▼
                ▼                                 [ Relevance Analysis Engine ]
    [ Scikit-learn Pipeline ]                     - TF-IDF Vector Space
   - Sublinear TF-IDF (1-2 n-grams)               - Cosine Similarity computation
   - Logistic Regression classifier               - Heuristic Threshold Categorization:
   - Output: LIKELY REAL / FAKE                     * Relevant (>= 0.10)
   - Feature Weight Explainability                  * Possibly Relevant (>= 0.03)
                │                                   * Not Relevant (< 0.03)
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                         [ Comprehensive Result Cards ]
                           A. ML Classification (Confidence %)
                           B. Fact-Check Status & Publisher Rating
                           C. Passage Relevance & Overlap N-Grams
                           D. Grounded Synthesis & Reasonings
                           E. Authoritative Citation Links
                                        │
                                        ▼
                           [ SQLite Database Engine ]
                           - Parameterized audit logging
                           - Dashboard distribution analytics
```

---

## 3. Technology Stack

- **Backend & Logic:** Python 3.11+
- **Frontend / UI:** Streamlit with bespoke Ink + Acid Lime CSS design system (`assets/style.css`)
- **Machine Learning & NLP:**
  - `scikit-learn`: `Pipeline`, `TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `metrics`
  - `numpy`, `pandas`: Data preprocessing and array transformations
  - `joblib`: Model serialization and persistence
- **Database:** SQLite (`sqlite3` standard library module) with context managers
- **Evidence Retrieval:** `requests`, `python-dotenv`, Google Fact Check Tools API

---

## 4. Project File Structure

```
ClaimLens/
├── app.py                      # Streamlit application with Ink + Acid Lime UI
├── train_model.py              # ML training pipeline, evaluation & explainability
├── database.py                 # SQLite database layer with parameterized queries
├── factcheck.py                # Live API + Curated local evidence matching
├── relevance.py                # TF-IDF + Cosine similarity passage relevance engine
├── utils.py                    # Text cleaning, validation, negation conflict detection
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows one-click startup batch script
├── .env.example                # Template for optional Google API key
├── .gitignore                  # Git ignore rules
├── README.md                   # Complete academic documentation & viva notes
│
├── assets/
│   └── style.css               # Ink + Acid Lime bespoke stylesheet
│
├── data/
│   ├── README.md               # Dataset sourcing & external import instructions
│   ├── train_dataset.csv       # Balanced real/fake training dataset
│   ├── curated_evidence.csv    # 12 documented fact-checks with passage pairs
│   └── sample_articles.csv     # Preloaded demonstration scenarios
│
├── models/
│   ├── claimlens_model.joblib  # Serialized Scikit-learn Pipeline
│   └── model_metadata.json     # Test split metrics and top feature weights
│
└── tests/
    ├── __init__.py
    ├── test_database.py        # SQLite unit tests (insert, search, clear, stats)
    ├── test_relevance.py       # Relevance engine unit tests (thresholds)
    └── test_inference.py       # NLP preprocessing, negation & model inference
```

---

## 5. Quickstart & Windows Execution Commands

### Step 1: Open Terminal in the Project Directory
```powershell
cd C:\Users\sharm\.gemini\antigravity\scratch\ClaimLens
```

### Step 2: Install Dependencies
```powershell
py -3.11 -m pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model
```powershell
py -3.11 train_model.py
```
*Evaluates the model on a held-out test split, prints accuracy, precision, recall, F1-score, and confusion matrix, and saves `models/claimlens_model.joblib`.*

### Step 4: Run Automated Tests
```powershell
py -3.11 -m unittest discover tests
```
*Runs all 13 automated unit tests validating database persistence, relevance calculations, and inference.*

### Step 5: Launch the Application
Double-click `run.bat` or run:
```powershell
py -3.11 -m streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 6. Preloaded Demonstration Scenarios

ClaimLens includes 3 preloaded demonstration scenarios directly accessible via one-click buttons on the **Analyze News** page:

### Scenario A: Likely Real News
- **Input Headline:** *"The Indian Space Research Organisation (ISRO) officially confirmed that the Chandrayaan-3 Vikram lander achieved a successful soft touchdown near the lunar south pole on August 23, 2023..."*
- **ML Classification:** **LIKELY REAL** (Confidence ~52%)
- **Fact-Check Result:** **PUBLISHED FACT-CHECK FOUND** (ISRO / BBC — Rating: `True`)
- **Relevance Analysis:** Direct touchdown passages classified as **RELEVANT** (Similarity ~35%), while Moon orbit trivia is classified as **NOT RELEVANT** (Similarity 0.0%).
- **Takeaway:** Demonstrates real news identification paired with authoritative source confirmation.

### Scenario B: Likely Fake Hoax
- **Input Headline:** *"SHOCKING: NASA scientists secretly leak classified dossier proving Earth will plunge into fifteen days of complete blackout darkness this November! Elite government officials are fleeing to underground survival bunkers while state media suppresses the catastrophic planetary alignment between Venus and Jupiter!"*
- **ML Classification:** **LIKELY FAKE** (Confidence ~61%)
- **Fact-Check Result:** **PUBLISHED FACT-CHECK FOUND** (Snopes — Rating: `False`)
- **Feature Weights:** Highlights sensationalist tokens such as *"shocking"*, *"secret"*, *"leak"*, and *"bunkers"*.
- **Takeaway:** Demonstrates detection of viral internet hoaxes backed by independent debunking archives.

### Scenario C: Unverified Local News
- **Input Headline:** *"The Springfield town municipality voted 4-2 on Tuesday evening to replace thirty sodium vapor street lamps along the north avenue with energy-efficient blue-spectrum LED panels..."*
- **ML Classification:** **LIKELY REAL** (Objective journalistic phrasing)
- **Fact-Check Result:** **UNVERIFIED** (No matching third-party fact-check found)
- **Takeaway:** Demonstrates the crucial academic distinction: **Unverified does not mean false**. Absence of a fact-check simply means third-party fact-checkers haven't audited this hyperlocal news.

---

## 7. Two-Minute Live Demonstration Script

> *"Good morning respected evaluators. Today I am presenting **ClaimLens**, a machine learning-based fake news detection and fact verification system.*
>
> *Most existing fake news projects make a critical mistake: they treat machine learning predictions as absolute truth. But writing style is not physical reality. A real emergency alert might be written hurriedly, while a fake conspiracy theory can be written in formal academic English.*
>
> *ClaimLens tackles this by combining three distinct layers:*
> *1. **First, Machine Learning:** We built a Scikit-learn Pipeline with TF-IDF Vectorization and Logistic Regression that analyzes linguistic features and outputs a calibrated confidence score. When we inspect the model weights, we can see positive weights for formal terms like 'domestic', 'percent', and 'federal', and negative weights for sensationalist words like 'shocking', 'secret', and 'leaked'.*
> *2. **Second, Published Fact-Checks:** ClaimLens queries external fact-checking databases—either live via the Google Fact Check Tools API or through our offline curated database of Reuters, AP, and Snopes investigations. In Scenario A, it matches ISRO's Chandrayaan-3 landing and returns the verified source URL.*
> *3. **Third, Evidence Relevance Detection:** Here is where ClaimLens stands out: it uses TF-IDF and Cosine Similarity to compare the claim against evidence passages. It categorizes them into Relevant, Possibly Relevant, and Not Relevant. As you can see, our relevance engine correctly identifies that a passage discussing lunar orbits is unrelated to the landing touchdown.*
> *4. **Finally, History and Analytics:** Every analysis is stored in a local SQLite database using secure parameterized queries. In the Dashboard tab, our analytics reflect real database records without fabricated metrics.*
>
> *Thank you. I welcome your questions."*

---

## 8. Viva Preparation & Theoretical Foundations

### Q1: How does TF-IDF work?
**Answer:** TF-IDF stands for *Term Frequency-Inverse Document Frequency*. It converts raw text into numerical feature vectors by multiplying two values:
- **Term Frequency (TF):** How frequently a word appears in a specific document.
- **Inverse Document Frequency (IDF):** The logarithm of the total number of documents divided by the number of documents containing that word.
Common words like *"the"* or *"and"* appear everywhere and receive an IDF close to zero, whereas informative domain words like *"chemotherapy"* or *"lander"* receive high weights. In ClaimLens, we use sublinear TF scaling (`sublinear_tf=True`) to replace raw frequency $tf$ with $1 + \log(tf)$ to prevent repetitive terms from dominating.

### Q2: Why did you choose Logistic Regression over a Deep Neural Network?
**Answer:** For a second-year mini project focused on tabular/n-gram NLP, Logistic Regression provides several key advantages:
1. **Explainability:** Logistic Regression coefficients directly map to feature importance. A positive coefficient indicates words correlated with legitimate journalism, while negative coefficients indicate sensationalism.
2. **Speed & Efficiency:** It trains in seconds and executes inference in milliseconds on a standard Windows laptop without requiring a GPU.
3. **Calibrated Probabilities:** The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputs smooth probabilities between 0 and 1.
4. **Appropriate Complexity:** Deep neural networks (e.g. BERT) easily overfit small benchmark datasets and act as unexplainable black boxes.

### Q3: What is Cosine Similarity and how is it used in Evidence Relevance?
**Answer:** Cosine similarity measures the cosine of the angle between two multi-dimensional vectors in vector space:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
It ranges from 0 (completely orthogonal / unrelated) to 1 (identical orientation). In ClaimLens, we transform both the claim and candidate evidence passages into TF-IDF vectors and calculate their cosine similarity. We use heuristic thresholds: $\ge 0.10$ for **Relevant**, $\ge 0.03$ for **Possibly Relevant**, and $< 0.03$ for **Not Relevant**.

### Q4: Why is an "Unverified" result necessary instead of just predicting True or False?
**Answer:** In scientific journalism and fact-checking, **absence of evidence is not evidence of absence**. If a user submits a local municipal news item that fact-checkers have never audited, giving a binary "False" prediction would be misleading and harmful. An "Unverified" state transparently informs the user that no documented third-party record exists.

### Q5: What is the risk of train-test data leakage, and how did you prevent it?
**Answer:** Data leakage occurs when information from the test dataset contaminates the training process, resulting in falsely inflated accuracy. We prevented leakage by:
1. Deduplicating all text records before splitting.
2. Performing a stratified train-test split before fitting the vectorizer.
3. Fitting the TF-IDF vocabulary **strictly on the training split only** using Scikit-learn's `Pipeline`.

---

## 9. GitHub-Ready Project Description

```markdown
### ClaimLens — Evidence-Assisted Fake News Detection System

ClaimLens is an open-source Python verification platform that combats digital misinformation by combining machine learning text classification with external fact-check retrieval and cosine-similarity evidence relevance detection.

**Key Features:**
- 🧠 **Scikit-learn Pipeline:** TF-IDF + Logistic Regression with explainable feature weights.
- 🔍 **Fact-Checking Layer:** Google Fact Check Tools API + Local Curated Evidence Repository (Reuters, Snopes, AP News).
- 📐 **Relevance Engine:** Cosine similarity scoring separating relevant citations from off-topic context.
- 🗄️ **SQLite Persistence:** Full analysis audit log with search, deletion, and real-time dashboard analytics.
- 🎨 **Ink + Acid Lime UI:** Modern editorial design system implemented in Streamlit.
```
