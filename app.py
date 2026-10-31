"""
ClaimLens — Machine Learning-Based Fake News Detection and Fact Verification System
Tagline: Think Before You Trust.
Visual Identity: Ink + Acid Lime (Investigative Journalism x Contemporary Tech)
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import streamlit as st

# Local modules
from database import (
    clear_history,
    delete_analysis,
    get_analysis,
    get_dashboard_stats,
    get_history,
    init_db,
    save_analysis,
)
from factcheck import FactCheckService
from relevance import EvidenceRelevanceEngine
from utils import clean_text, extract_key_terms, get_text_statistics, validate_input

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "claimlens_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")
SAMPLES_CSV = os.path.join(BASE_DIR, "data", "sample_articles.csv")

st.set_page_config(
    page_title="ClaimLens — Think Before You Trust",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize SQLite database schema
init_db()


# --- CSS Injection ---
def load_custom_css():
    """Inject custom Ink + Acid Lime stylesheet."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline theme definition if file is moved
        st.markdown(
            """
            <style>
            .stApp { background-color: #161616; color: #F2EEE5; }
            </style>
            """,
            unsafe_allow_html=True,
        )


load_custom_css()


# --- Model & Service Caching ---
@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Load serialized Joblib pipeline and metadata JSON."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        metadata = None
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        return model, metadata
    except Exception as e:
        return None, None


@st.cache_resource(show_spinner=False)
def get_factcheck_service():
    """Instantiate FactCheckService."""
    return FactCheckService()


@st.cache_resource(show_spinner=False)
def get_relevance_engine():
    """Instantiate EvidenceRelevanceEngine."""
    return EvidenceRelevanceEngine()


# --- Session State Management ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Analyze News"
if "claim_input_text" not in st.session_state:
    st.session_state.claim_input_text = ""
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# --- Header & Brand Component ---
def render_brand_header():
    """Render consistent minimalist editorial wordmark."""
    st.markdown(
        """
        <div class="claimlens-brand">
            <div class="lens-icon">⌕</div>
            <div>
                <h1 class="brand-title">CLAIMLENS</h1>
                <div class="brand-tagline">Think Before You Trust • Academic Verification Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# PAGE 1: HOME
# ==========================================
def render_home_page():
    render_brand_header()

    st.markdown('<span class="eyebrow">Investigative News Intelligence</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="font-size: 2.8rem; margin-top: 0.2rem; margin-bottom: 1.2rem; line-height: 1.15;">
            Don’t just read it. <span style="color: #DFFF62;">Question it.</span>
        </h2>
        <p style="font-size: 1.15rem; max-width: 760px; color: #B8B8B2; margin-bottom: 2rem;">
            ClaimLens is an evidence-assisted fact verification platform designed to combat modern misinformation.
            It pairs Scikit-learn linguistic classification with evidence retrieval and passage relevance scoring
            to separate stylistic sensationalism from documented factual proof.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Start Analyzing News →", use_container_width=True):
            st.session_state.current_page = "Analyze News"
            st.session_state.sidebar_navigation = "Analyze News"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="editorial-hr">', unsafe_allow_html=True)

    # How it Works Section
    st.markdown('<span class="eyebrow">Methodology</span>', unsafe_allow_html=True)
    st.markdown("### How ClaimLens Works")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="lens-card">
                <span class="step-number">STEP 01</span>
                <h4 style="color: #F2EEE5; margin-top: 0.5rem; margin-bottom: 0.4rem;">Paste & Preprocess</h4>
                <p class="text-muted">
                    Submit any suspicious news headline, article excerpt, or forwarded claim. Text is normalized
                    and validated to prepare for n-gram feature extraction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="lens-card">
                <span class="step-number">STEP 02</span>
                <h4 style="color: #F2EEE5; margin-top: 0.5rem; margin-bottom: 0.4rem;">ML Pattern Analysis</h4>
                <p class="text-muted">
                    A genuine Scikit-learn TF-IDF + Logistic Regression model inspects linguistic markers,
                    sensationalist hyperbole, and tone to generate a stylistic classification estimate.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="lens-card">
                <span class="step-number">STEP 03</span>
                <h4 style="color: #F2EEE5; margin-top: 0.5rem; margin-bottom: 0.4rem;">Evidence Verification</h4>
                <p class="text-muted">
                    ClaimLens queries verified fact-checking records (Reuters, AP, Snopes) and runs cosine similarity
                    relevance checks to differentiate relevant evidence from off-topic context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Core Academic Distinction Notice
    st.markdown(
        """
        <div class="lens-card-elevated" style="margin-top: 1.5rem; border-left: 3px solid #DFFF62;">
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #DFFF62; text-transform: uppercase; margin-bottom: 0.3rem;">
                Academic Distinction Notice
            </div>
            <strong style="color: #F2EEE5;">Machine Learning Prediction ≠ Factual Truth</strong>
            <p style="color: #B8B8B2; font-size: 0.92rem; margin-top: 0.3rem; margin-bottom: 0;">
                An ML text classifier detects statistical writing styles (such as emotional clickbait or formal journalism).
                However, a true story can be poorly written, and a fabricated rumor can mimic journalistic grammar.
                ClaimLens never treats ML scores as proof of reality—factual confirmation requires genuine external citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


# ==========================================
# PAGE 2: ANALYZE NEWS
# ==========================================
def render_analyze_page():
    render_brand_header()

    st.markdown('<span class="eyebrow">Investigation Workspace</span>', unsafe_allow_html=True)
    st.markdown(
        '<h2 style="margin-top: 0.1rem; margin-bottom: 0.6rem;">Analyze News Credibility</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="text-muted">Enter a news headline or full article text below to evaluate linguistic patterns, retrieve published fact-checks, and inspect passage relevance.</p>',
        unsafe_allow_html=True,
    )

    # Preloaded Demonstrations Row
    st.markdown('<span class="eyebrow" style="font-size: 0.68rem;">Quick Demonstration Scenarios</span>', unsafe_allow_html=True)
    d_col1, d_col2, d_col3, d_col4 = st.columns([1.2, 1.2, 1.2, 0.8])

    with d_col1:
        if st.button("🟢 Scenario A: Likely Real", use_container_width=True):
            st.session_state.claim_input_text = (
                "The Indian Space Research Organisation (ISRO) officially confirmed that the "
                "Chandrayaan-3 Vikram lander achieved a successful soft touchdown near the lunar "
                "south pole on August 23, 2023. Prime Minister Narendra Modi and global space "
                "agencies congratulated the mission scientists. The Pragyan rover rolled out to "
                "begin chemical analysis of the lunar soil."
            )
            st.rerun()

    with d_col2:
        if st.button("🔴 Scenario B: Likely Fake", use_container_width=True):
            st.session_state.claim_input_text = (
                "SHOCKING: NASA scientists secretly leak classified dossier proving Earth will plunge "
                "into fifteen days of complete blackout darkness this November! Elite government officials "
                "are fleeing to underground survival bunkers while state media suppresses the catastrophic "
                "planetary alignment between Venus and Jupiter!"
            )
            st.rerun()

    with d_col3:
        if st.button("🟡 Scenario C: Unverified", use_container_width=True):
            st.session_state.claim_input_text = (
                "The Springfield town municipality voted 4-2 on Tuesday evening to replace thirty "
                "sodium vapor street lamps along the north avenue with energy-efficient blue-spectrum "
                "LED panels. Residents attended the public meeting to voice minor concerns regarding "
                "neighborhood glare, but the budget committee approved the small procurement grant."
            )
            st.rerun()

    with d_col4:
        if st.button("Clear Input", use_container_width=True):
            st.session_state.claim_input_text = ""
            st.session_state.last_analysis = None
            st.rerun()

    # User Input Text Area
    user_text = st.text_area(
        "News Text or Article Body",
        value=st.session_state.claim_input_text,
        height=160,
        placeholder="Paste a news story, social media post, or headline here (minimum 3 words / 15 characters)...",
    )

    # Dynamic Text Statistics
    stats = get_text_statistics(user_text)
    stat_col1, stat_col2, stat_col3 = st.columns([1, 1, 3])
    with stat_col1:
        st.markdown(f'<span class="text-muted">Words: <strong>{stats["word_count"]}</strong></span>', unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f'<span class="text-muted">Characters: <strong>{stats["char_count"]}</strong></span>', unsafe_allow_html=True)
    with stat_col3:
        pass

    # Action Button
    st.markdown('<div class="primary-btn" style="margin-top: 0.8rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Run Full ClaimLens Analysis", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Trigger Analysis
    if analyze_clicked:
        is_valid, err_msg = validate_input(user_text)
        if not is_valid:
            st.warning(f"⚠️ {err_msg}")
            return

        with st.spinner("Analyzing linguistic patterns and retrieving evidence..."):
            model, metadata = load_ml_model()
            factcheck_service = get_factcheck_service()
            relevance_engine = get_relevance_engine()

            # 1. ML Prediction
            ml_pred = "MODEL UNAVAILABLE"
            ml_score = None
            ml_score_pct = None
            top_matching_features = []

            if model is not None:
                cleaned = clean_text(user_text)
                pred_label = model.predict([cleaned])[0]
                proba = model.predict_proba([cleaned])[0]

                # Label 1 is Real, 0 is Fake
                if pred_label == 1:
                    ml_pred = "LIKELY REAL"
                    ml_score = float(proba[1])
                else:
                    ml_pred = "LIKELY FAKE"
                    ml_score = float(proba[0])

                ml_score_pct = round(ml_score * 100, 1)

                # Find which vocabulary words in input affected the score
                try:
                    vec = model.named_steps["tfidf"]
                    clf = model.named_steps["clf"]
                    feat_names = vec.get_feature_names_out()
                    coefs = clf.coef_[0]
                    words_in_input = set(clean_text(user_text).lower().split())
                    for term, coef in zip(feat_names, coefs):
                        if term in words_in_input:
                            top_matching_features.append(
                                {"term": term, "coefficient": round(float(coef), 3)}
                            )
                    # Sort by absolute influence
                    top_matching_features.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
                    top_matching_features = top_matching_features[:6]
                except Exception:
                    top_matching_features = []
            else:
                ml_pred = "MODEL NOT TRAINED"

            # 2. Fact Verification
            fact_res = factcheck_service.verify(user_text)

            # 3. Evidence Relevance Analysis
            candidate_passages = fact_res.get("passages", [])
            evaluated_passages = []
            if candidate_passages:
                evaluated_passages = relevance_engine.evaluate_passages(
                    user_text, candidate_passages
                )

            # 4. Save to SQLite Database
            db_id = save_analysis(
                claim_text=user_text,
                model_prediction=ml_pred,
                model_score=ml_score,
                factcheck_status=fact_res.get("status", "UNVERIFIED"),
                factcheck_publisher=fact_res.get("publisher"),
                factcheck_url=fact_res.get("source_url"),
                relevance_summary=(
                    f"Evaluated {len(evaluated_passages)} passages"
                    if evaluated_passages
                    else "No candidate passages"
                ),
            )

            # Store result in session state
            st.session_state.last_analysis = {
                "id": db_id,
                "input_text": user_text,
                "ml_prediction": ml_pred,
                "ml_score": ml_score,
                "ml_score_pct": ml_score_pct,
                "metadata": metadata,
                "top_features": top_matching_features,
                "factcheck": fact_res,
                "evaluated_passages": evaluated_passages,
            }

    # Render Active Result
    if st.session_state.last_analysis:
        display_analysis_results(st.session_state.last_analysis)

    render_footer()


def display_analysis_results(result: Dict[str, Any]):
    """Render structured investigative results cards."""
    st.markdown('<hr class="editorial-hr">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Investigation Report</span>', unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="margin-top: 0.1rem; margin-bottom: 1.5rem;">Results for Claim #{result["id"]}</h3>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------
    # SECTION A: ML CLASSIFICATION
    # ---------------------------------------------
    ml_pred = result["ml_prediction"]
    ml_score_pct = result["ml_score_pct"]

    badge_class = "badge-unverified"
    if ml_pred == "LIKELY REAL":
        badge_class = "badge-real"
    elif ml_pred == "LIKELY FAKE":
        badge_class = "badge-fake"

    st.markdown(
        f"""
        <div class="lens-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.8rem;">
                <div>
                    <span class="eyebrow" style="font-size: 0.7rem; color: #B8B8B2;">SECTION A</span>
                    <h4 style="margin: 0; color: #F2EEE5;">Machine Learning Classification</h4>
                </div>
                <div>
                    <span class="badge {badge_class}">{ml_pred}</span>
                </div>
            </div>
            <div style="font-size: 1.15rem; margin-bottom: 0.6rem;">
                Confidence Estimate: <strong style="color: #DFFF62;">{ml_score_pct if ml_score_pct is not None else 'N/A'}%</strong>
            </div>
            <p class="text-muted" style="margin-bottom: 0.5rem;">
                The Scikit-learn Logistic Regression classifier evaluated the textual vocabulary against trained news writing patterns.
                This score represents <em>stylistic probability</em>, not an independent determination of physical truth.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("top_features"):
        with st.expander("🔍 View Influential Words Identified in Model Weights"):
            st.markdown(
                "<p class='text-muted'>Terms found in your input that carried weight in the model's decision:</p>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(result["top_features"]))
            for idx, feat in enumerate(result["top_features"]):
                with cols[idx]:
                    direction = "Real (+)" if feat["coefficient"] > 0 else "Fake (-)"
                    st.metric(label=feat["term"], value=direction, delta=f"{feat['coefficient']}")

    # ---------------------------------------------
    # SECTION B: FACT-CHECK EVIDENCE
    # ---------------------------------------------
    fc = result["factcheck"]
    fc_status = fc.get("status", "UNVERIFIED")
    fc_badge = "badge-unverified"
    rating_lower = str(fc.get("original_rating", "")).lower()
    if "FOUND" in fc_status:
        if any(k in rating_lower for k in ["true", "correct", "accurate"]):
            fc_badge = "badge-real"
        elif any(k in rating_lower for k in ["misleading", "partly", "context"]):
            fc_badge = "badge-possible"
        else:
            fc_badge = "badge-fake"

    publisher = fc.get("publisher") or "Not Identified"
    rating = fc.get("original_rating") or "No published rating"
    source_url = fc.get("source_url")
    date_checked = fc.get("date_checked") or "N/A"

    st.markdown(
        f"""
        <div class="lens-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.8rem;">
                <div>
                    <span class="eyebrow" style="font-size: 0.7rem; color: #B8B8B2;">SECTION B</span>
                    <h4 style="margin: 0; color: #F2EEE5;">Published Fact-Check Verification</h4>
                </div>
                <div>
                    <span class="badge {fc_badge}">{fc_status}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                <div class="lens-card-elevated" style="margin: 0;">
                    <span class="eyebrow" style="font-size: 0.65rem;">Publisher</span>
                    <div style="color: #F2EEE5; font-weight: 600;">{publisher}</div>
                </div>
                <div class="lens-card-elevated" style="margin: 0;">
                    <span class="eyebrow" style="font-size: 0.65rem;">Original Documented Rating</span>
                    <div style="color: #DFFF62; font-weight: 600;">{rating}</div>
                </div>
                <div class="lens-card-elevated" style="margin: 0;">
                    <span class="eyebrow" style="font-size: 0.65rem;">Verification Date</span>
                    <div style="color: #F2EEE5; font-weight: 600;">{date_checked}</div>
                </div>
            </div>
            <p style="color: #F2EEE5; font-size: 0.95rem; margin-bottom: 0.6rem;">
                {fc.get("summary", "")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------
    # SECTION C: EVIDENCE RELEVANCE ANALYSIS
    # ---------------------------------------------
    passages = result.get("evaluated_passages", [])
    st.markdown(
        """
        <div style="margin-top: 1.5rem; margin-bottom: 0.8rem;">
            <span class="eyebrow" style="font-size: 0.7rem; color: #B8B8B2;">SECTION C</span>
            <h4 style="margin: 0; color: #F2EEE5;">Evidence Passage Relevance Engine (TF-IDF & Cosine Similarity)</h4>
            <p class="text-muted" style="margin-top: 0.2rem; font-size: 0.88rem;">
                Passages are classified by topical relatedness using cosine distance.
                <strong style="color: #DFFF62;">High relevance denotes topical overlap, not factual truth.</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if passages:
        for p in passages:
            cls = p.get("classification", "NOT RELEVANT")
            badge = "badge-irrelevant"
            if cls == "RELEVANT":
                badge = "badge-relevant"
            elif cls == "POSSIBLY RELEVANT":
                badge = "badge-possible"

            terms_html = " ".join(
                [f"<span style='background:#161616; padding: 2px 6px; border-radius:3px; margin-right:4px; font-family: monospace; font-size:0.8rem;'>{t}</span>" for t in p.get("shared_terms", [])]
            )

            st.markdown(
                f"""
                <div class="lens-card-elevated" style="margin-bottom: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="badge {badge}">{cls}</span>
                        <span style="font-family: var(--font-mono); font-size: 0.85rem; color: #DFFF62;">
                            Cosine Similarity: {p.get('similarity_percentage')}% ({p.get('similarity_score')})
                        </span>
                    </div>
                    <div style="font-size: 0.96rem; color: #F2EEE5; margin-bottom: 0.5rem;">
                        "{p.get('text')}"
                    </div>
                    <div class="text-muted" style="font-size: 0.85rem; margin-bottom: 0.4rem;">
                        <strong>Engine Rationale:</strong> {p.get('explanation')}
                    </div>
                    {f'<div class="text-muted" style="font-size: 0.82rem;"><strong>Shared Overlap Terms:</strong> {terms_html}</div>' if terms_html else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("ℹ️ No specific evidence passages available for relevance scoring for this claim.")

    # ---------------------------------------------
    # SECTION D: SYSTEM EXPLANATION & REASONING
    # ---------------------------------------------
    st.markdown(
        f"""
        <div class="lens-card" style="margin-top: 1.5rem;">
            <span class="eyebrow" style="font-size: 0.7rem; color: #B8B8B2;">SECTION D</span>
            <h4 style="margin: 0; color: #F2EEE5; margin-bottom: 0.8rem;">Synthesis & Explanation</h4>
            <div style="color: #B8B8B2; font-size: 0.92rem; line-height: 1.65;">
                <p>
                    <strong>1. Model Prediction:</strong> The ML model classified this text as <strong>{ml_pred}</strong>
                    with an estimated classification confidence of {ml_score_pct if ml_score_pct is not None else 'N/A'}%.
                    This reflects linguistic markers such as vocabulary distribution, sensationalism indicators, and syntax.
                </p>
                <p>
                    <strong>2. Verification Context:</strong> External fact-checking records evaluated status as <strong>{fc_status}</strong>.
                    {'A verified published source was located addressing the core claim.' if 'FOUND' in fc_status else 'No documented third-party debunking or confirmation was found.'}
                </p>
                <p style="margin-bottom: 0;">
                    <strong>3. Assessment Recommendation:</strong> Always consult authoritative primary sources.
                    When evidence is Unverified, avoid treating absence of a fact-check as confirmation of authenticity.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------
    # SECTION E: SOURCES & REFERENCES
    # ---------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 1.25rem;">
            <span class="eyebrow" style="font-size: 0.7rem; color: #B8B8B2;">SECTION E</span>
            <h4 style="margin: 0; color: #F2EEE5; margin-bottom: 0.6rem;">Authoritative Sources & Citations</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if source_url:
        st.markdown(
            f"""
            <div class="lens-card-elevated" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #F2EEE5;">{publisher}</strong>
                    <div class="text-muted" style="font-size: 0.85rem;">Date Checked: {date_checked}</div>
                </div>
                <div>
                    <a href="{source_url}" target="_blank" style="color: #DFFF62; text-decoration: none; font-weight: 600; font-size: 0.9rem;">
                        View Documented Source Record ↗
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="lens-card-elevated text-muted" style="font-size: 0.9rem;">
                No external citation URL recorded for this input.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# PAGE 3: HISTORY
# ==========================================
def render_history_page():
    render_brand_header()

    st.markdown('<span class="eyebrow">Audit Log</span>', unsafe_allow_html=True)
    st.markdown('<h2>Analysis History</h2>', unsafe_allow_html=True)
    st.markdown('<p class="text-muted">Review, search, and manage previously analyzed news items stored in SQLite.</p>', unsafe_allow_html=True)

    # Search & Actions Bar
    c_search, c_clear = st.columns([3, 1])
    with c_search:
        search_query = st.text_input("Search history records", placeholder="Search by claim keywords or prediction...", label_visibility="collapsed")
    with c_clear:
        if st.button("Clear History", use_container_width=True):
            st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("Are you sure you want to delete all historical analysis records?")
        col_yes, col_no = st.columns([1, 4])
        with col_yes:
            if st.button("Yes, Clear All"):
                count = clear_history()
                st.session_state.confirm_clear = False
                st.session_state.last_analysis = None
                st.success(f"Cleared {count} historical records.")
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()

    # Fetch History
    records = get_history(limit=50, search_query=search_query)

    if not records:
        st.markdown(
            """
            <div class="lens-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem; color: #45453F;">⌕</div>
                <h4 style="color: #F2EEE5; margin-bottom: 0.3rem;">No Historical Records Found</h4>
                <p class="text-muted" style="max-width: 420px; margin: 0 auto;">
                    Run your first news analysis in the "Analyze News" section to begin building a local audit log.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_footer()
        return

    st.markdown(f'<span class="text-muted">Displaying <strong>{len(records)}</strong> saved record(s).</span>', unsafe_allow_html=True)

    for rec in records:
        pred = rec["model_prediction"]
        badge_cls = "badge-unverified"
        if pred == "LIKELY REAL":
            badge_cls = "badge-real"
        elif pred == "LIKELY FAKE":
            badge_cls = "badge-fake"

        # Truncate claim text for clean presentation
        text_preview = rec["claim_text"]
        if len(text_preview) > 130:
            text_preview = text_preview[:127] + "..."

        with st.container():
            st.markdown(
                f"""
                <div class="lens-card" style="margin-bottom: 0.8rem; padding: 1.1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.4rem;">
                        <span class="badge {badge_cls}">{pred}</span>
                        <span style="font-family: var(--font-mono); font-size: 0.78rem; color: #B8B8B2;">{rec['created_at']}</span>
                    </div>
                    <div style="font-size: 0.98rem; font-weight: 500; color: #F2EEE5; margin-bottom: 0.4rem;">
                        "{text_preview}"
                    </div>
                    <div style="display: flex; gap: 1rem; font-size: 0.82rem; color: #B8B8B2;">
                        <span>Confidence: <strong style="color: #DFFF62;">{round(rec['model_score']*100, 1) if rec['model_score'] is not None else 'N/A'}%</strong></span>
                        <span>Fact-Check: <strong>{rec['factcheck_status']}</strong></span>
                        {f"<span>Publisher: {rec['factcheck_publisher']}</span>" if rec['factcheck_publisher'] else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_view, col_del = st.columns([1, 1])
            with col_view:
                with st.expander(f"Inspect Full Text #{rec['id']}"):
                    st.write(rec["claim_text"])
                    if rec["factcheck_url"]:
                        st.markdown(f"[View Verified Citation ↗]({rec['factcheck_url']})")
            with col_del:
                if st.button(f"Delete Record #{rec['id']}", key=f"del_{rec['id']}"):
                    delete_analysis(rec["id"])
                    st.rerun()

    render_footer()


# ==========================================
# PAGE 4: DASHBOARD
# ==========================================
def render_dashboard_page():
    render_brand_header()

    st.markdown('<span class="eyebrow">Metrics & Distribution</span>', unsafe_allow_html=True)
    st.markdown('<h2>Investigation Analytics</h2>', unsafe_allow_html=True)
    st.markdown('<p class="text-muted">Live metrics calculated directly from the SQLite database of executed analyses.</p>', unsafe_allow_html=True)

    stats = get_dashboard_stats()

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Analyses", value=stats["total_analyses"])
    with m2:
        st.metric(label="Likely Real", value=stats["likely_real_count"])
    with m3:
        st.metric(label="Likely Fake", value=stats["likely_fake_count"])
    with m4:
        st.metric(label="Unverified Cases", value=stats["unverified_count"])

    st.markdown('<hr class="editorial-hr">', unsafe_allow_html=True)

    # Prediction Distribution Chart
    st.markdown('### Prediction Distribution')
    if stats["total_analyses"] > 0:
        chart_data = pd.DataFrame(
            {
                "Category": ["Likely Real", "Likely Fake", "Unverified"],
                "Count": [
                    stats["likely_real_count"],
                    stats["likely_fake_count"],
                    stats["unverified_count"],
                ],
            }
        ).set_index("Category")
        st.bar_chart(chart_data, color="#DFFF62")
    else:
        st.info("No analyses recorded yet. Start investigating claims in the Analyze News tab to generate distribution charts.")

    st.markdown('<hr class="editorial-hr">', unsafe_allow_html=True)

    # Recent Activity Feed
    st.markdown('### Recently Evaluated Headlines')
    recent = stats.get("recent_analyses", [])
    if recent:
        for r in recent:
            excerpt = r["claim_text"]
            if len(excerpt) > 100:
                excerpt = excerpt[:97] + "..."
            p_color = "#58D68D" if r["model_prediction"] == "LIKELY REAL" else "#FF5A5A"
            st.markdown(
                f"""
                <div class="lens-card-elevated" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div>
                        <span style="font-weight: 600; color: #F2EEE5;">"{excerpt}"</span>
                        <div class="text-muted" style="font-size: 0.8rem;">Saved: {r['created_at']}</div>
                    </div>
                    <div>
                        <span style="font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; color: {p_color};">
                            {r['model_prediction']}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<p class="text-muted">No recent items recorded.</p>', unsafe_allow_html=True)

    render_footer()


# ==========================================
# PAGE 5: ABOUT
# ==========================================
def render_about_page():
    render_brand_header()

    st.markdown('<span class="eyebrow">Academic Background & Architecture</span>', unsafe_allow_html=True)
    st.markdown('<h2>About ClaimLens</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 1.1rem; color: #B8B8B2; margin-bottom: 1.5rem;">
            ClaimLens was developed as a Second-Year B.Tech Computer Science Python Mini Project.
            It addresses the rapid proliferation of digital misinformation by bridging the gap
            between statistical Natural Language Processing and empirical evidence retrieval.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Core Concepts
    st.markdown(
        """
        <div class="lens-card">
            <h4 style="color: #DFFF62; margin-bottom: 0.6rem;">Why Linguistic Classification Isn’t Enough</h4>
            <p style="color: #F2EEE5; font-size: 0.95rem; line-height: 1.6;">
                Most basic fake news detectors rely entirely on text classification. If a user pastes an article,
                the model computes word frequencies and classifies it as True or False based on writing style.
                However, <strong>writing style is not equivalent to factual truth</strong>. A completely fabricated conspiracy
                story can be phrased in formal academic English, while a genuine breaking disaster report might use
                informal or urgent language.
            </p>
            <p style="color: #F2EEE5; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0;">
                ClaimLens solves this by combining <strong>pattern-based machine learning</strong> with an
                <strong>evidence-assisted fact-checking layer</strong> and a <strong>cosine similarity relevance engine</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tech Stack Cards
    st.markdown("### Technical Architecture & Mathematical Foundations")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            """
            <div class="lens-card-elevated">
                <span class="eyebrow">Feature Extraction</span>
                <h4 style="color: #F2EEE5; margin-top: 0.3rem;">TF-IDF Vectorizer</h4>
                <p class="text-muted" style="font-size: 0.88rem;">
                    Term Frequency-Inverse Document Frequency reflects how important a word or n-gram is to a text within a collection.
                    Rare, highly informative words receive higher weights, while ubiquitous stop words are suppressed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="lens-card-elevated">
                <span class="eyebrow">Classification</span>
                <h4 style="color: #F2EEE5; margin-top: 0.3rem;">Logistic Regression</h4>
                <p class="text-muted" style="font-size: 0.88rem;">
                    A linear model that applies the sigmoid function to a weighted linear combination of TF-IDF features,
                    producing a calibrated probability estimate between 0 (Fake) and 1 (Real).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """
            <div class="lens-card-elevated">
                <span class="eyebrow">Evidence Scoring</span>
                <h4 style="color: #F2EEE5; margin-top: 0.3rem;">Cosine Similarity</h4>
                <p class="text-muted" style="font-size: 0.88rem;">
                    Calculates the cosine of the angle between high-dimensional TF-IDF vectors of a claim and candidate evidence passages,
                    quantifying topical overlap independent of document length.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model Evaluation Metrics
    st.markdown('<hr class="editorial-hr">', unsafe_allow_html=True)
    st.markdown("### Measured Machine Learning Performance")
    _, metadata = load_ml_model()
    if metadata and "metrics" in metadata:
        m = metadata["metrics"]
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1:
            st.metric("Test Accuracy", f"{m['accuracy']*100:.1f}%")
        with cm2:
            st.metric("Test Precision", f"{m['precision']:.3f}")
        with cm3:
            st.metric("Test Recall", f"{m['recall']:.3f}")
        with cm4:
            st.metric("Test F1-Score", f"{m['f1_score']:.3f}")
        st.markdown(
            f"""
            <div class="text-muted" style="font-size: 0.85rem; margin-top: 0.5rem;">
                Model trained on {metadata.get('train_samples', 0)} samples and validated on a held-out test split of {metadata.get('test_samples', 0)} samples.
                Trained at: {metadata.get('training_timestamp', 'Unknown')}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Model metadata not currently available. Please execute 'py -3.11 train_model.py' to generate.")

    # Limitations & Scope
    st.markdown('<hr class="editorial-hr">', unsafe_allow_html=True)
    st.markdown("### System Limitations & Future Scope")
    st.markdown(
        """
        <div class="lens-card">
            <h4 style="color: #F2EEE5; margin-bottom: 0.5rem;">Documented Limitations</h4>
            <ul style="color: #B8B8B2; font-size: 0.92rem; padding-left: 1.2rem; margin-bottom: 1rem;">
                <li><strong>Distribution Shift:</strong> Models trained on specific news cycles may lose accuracy when evaluating emerging contemporary slang or novel events.</li>
                <li><strong>Absence of Evidence:</strong> An 'Unverified' status indicates lack of documented fact-checks, not proof of falsehood.</li>
                <li><strong>Semantic Nuance:</strong> TF-IDF does not preserve long-distance syntactic dependencies or subtle sarcasm.</li>
            </ul>
            <h4 style="color: #F2EEE5; margin-bottom: 0.5rem;">Future Enhancements</h4>
            <ul style="color: #B8B8B2; font-size: 0.92rem; padding-left: 1.2rem; margin-bottom: 0;">
                <li>Automated multi-claim extraction from long investigative articles.</li>
                <li>Transformer-based natural language inference (NLI) for automated stance detection (Supports vs Refutes).</li>
                <li>Cross-lingual claim verification for multilingual WhatsApp forwards.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


# --- Shared Footer Component ---
def render_footer():
    """Render consistent editorial footer."""
    st.markdown(
        """
        <div class="lens-footer">
            <div>
                <strong>CLAIMLENS</strong> • Academic Mini Project 2026
            </div>
            <div>
                Think Before You Trust • Designed with Ink + Acid Lime
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Navigation Routing ---
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="font-family: var(--font-editorial); font-size: 1.4rem; font-weight: 800; color: #F2EEE5;">
                    CLAIM<span style="color: #DFFF62;">LENS</span>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: #DFFF62; text-transform: uppercase;">
                    Fact Verification System
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "sidebar_navigation" not in st.session_state:
            st.session_state.sidebar_navigation = st.session_state.current_page

        nav_selection = st.radio(
            "Navigation",
            ["Home", "Analyze News", "History", "Dashboard", "About"],
            key="sidebar_navigation",
            label_visibility="collapsed",
        )
        st.session_state.current_page = nav_selection

        st.markdown('<hr class="editorial-hr" style="margin: 1.25rem 0;">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow" style="font-size: 0.68rem;">System Status</span>', unsafe_allow_html=True)

        # Check model status
        model, _ = load_ml_model()
        if model is not None:
            st.markdown("🟢 <span style='font-size: 0.85rem; color: #58D68D;'>ML Model Ready</span>", unsafe_allow_html=True)
        else:
            st.markdown("🟡 <span style='font-size: 0.85rem; color: #E0C870;'>Model Not Loaded</span>", unsafe_allow_html=True)

        # Check API status
        api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
        if api_key:
            st.markdown("🟢 <span style='font-size: 0.85rem; color: #58D68D;'>Google Fact Check Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("⚪ <span style='font-size: 0.85rem; color: #B8B8B2;'>Curated Local Mode</span>", unsafe_allow_html=True)

        st.markdown('<hr class="editorial-hr" style="margin: 1.25rem 0;">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #B8B8B2; line-height: 1.4;">
                B.Tech CS Mini Project<br>
                Target Demo: Oct 27, 2026<br>
                Python 3.11 • Streamlit • Scikit-learn
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route to selected page
    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Analyze News":
        render_analyze_page()
    elif st.session_state.current_page == "History":
        render_history_page()
    elif st.session_state.current_page == "Dashboard":
        render_dashboard_page()
    elif st.session_state.current_page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
