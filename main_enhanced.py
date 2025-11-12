# ============================================
# 🌈 Streamlit NLP Phase-wise Model Comparison with Google Fact Check
# ============================================

import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import requests
import time

# ============================
# NLTK Setup
# ============================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================
# Google Fact Check API Key (HARDCODED)
# ============================
# Replace 'YOUR_API_KEY_HERE' with your actual Google Fact Check API key
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # <-- PASTE YOUR API KEY HERE

# ============================
# Google Fact Check API Integration (NEW)
# ============================
def check_fact_with_google(claim_text, api_key):
    """
    Query Google Fact Check API for a given claim.
    Returns 1 for TRUE, 0 for FALSE, and None if no data is available.
    """
    if not api_key or api_key.strip() == "":
        return None
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": api_key,
        "query": claim_text,
        "languageCode": "en"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "claims" in data and len(data["claims"]) > 0:
                first_claim = data["claims"][0]
                if "claimReview" in first_claim and len(first_claim["claimReview"]) > 0:
                    rating = first_claim["claimReview"][0].get("textualRating", "").lower()
                    
                    # Map ratings to binary
                    true_ratings = ["true", "mostly true", "correct", "accurate", "verified"]
                    false_ratings = ["false", "mostly false", "incorrect", "inaccurate", "pants on fire"]
                    
                    if any(tr in rating for tr in true_ratings):
                        return 1
                    elif any(fr in rating for fr in false_ratings):
                        return 0
        return None
    except Exception:
        return None

def get_google_fact_check_labels(texts, test_indices, api_key, progress_bar=None):
    """
    Get fact check labels ONLY for test set statements (much faster!)
    """
    labels = [None] * len(texts)
    
    # Only check test set statements
    for idx, test_idx in enumerate(test_indices):
        text = texts[test_idx]
        label = check_fact_with_google(text, api_key)
        labels[test_idx] = label
        
        if progress_bar:
            progress_bar.progress((idx + 1) / len(test_indices))
        
        time.sleep(0.2)  # Small delay to avoid rate limiting
    
    return labels

# ============================
# Feature Extractors (ORIGINAL - UNCHANGED)
# ============================
def lexical_preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

def syntactic_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return " ".join([tag for word, tag in pos_tags])

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    sents = sent_tokenize(text)
    return f"{len(sents)} {' '.join([s.split()[0] for s in sents if s])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Model Evaluation (ORIGINAL - UNCHANGED)
# ============================
def evaluate_models(X_features, y):
    results = {}
    trained_models = {}  # NEW: Store trained models for Google Fact Check
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # NEW: Get test indices for Google Fact Check
    if hasattr(y, 'index'):
        test_indices = y_test.index.tolist()
    else:
        # For cases where y doesn't have an index (like numpy arrays)
        all_indices = list(range(len(y)))
        train_size = len(y_train)
        test_indices = all_indices[train_size:]
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
            trained_models[name] = (model, X_test, y_test)  # NEW: Store for later use
        except Exception as e:
            results[name] = f"Error: {str(e)}"
            trained_models[name] = None
    
    return results, trained_models, test_indices  # NEW: Return additional data

# ============================
# Google Fact Check Evaluation (NEW)
# ============================
def evaluate_with_google_labels(trained_models, google_labels, test_indices):
    """
    Evaluate models using Google Fact Check labels as ground truth
    """
    results = {}
    
    for name, model_data in trained_models.items():
        if model_data is None:
            results[name] = "N/A"
            continue
        
        model, X_test, y_test = model_data
        
        # Get Google labels for test indices
        google_test_labels = [google_labels[i] for i in test_indices]
        
        # Filter out None values (statements without Google data)
        valid_indices = [i for i, label in enumerate(google_test_labels) if label is not None]
        
        if len(valid_indices) == 0:
            results[name] = "No data"
            continue
        
        # Get predictions for valid samples
        if hasattr(X_test, 'iloc'):
            X_test_valid = X_test.iloc[valid_indices]
        else:
            X_test_valid = X_test[valid_indices]
        
        google_test_valid = [google_test_labels[i] for i in valid_indices]
        
        try:
            y_pred = model.predict(X_test_valid)
            acc = accuracy_score(google_test_valid, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception:
            results[name] = "Error"
    
    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase-wise Analysis", layout="wide")
st.title("🧠 NLP Phase-wise Analysis with Google Fact Check")
st.markdown(
    "<p style='color:gray;'>Upload a dataset, choose an NLP phase, and compare multiple ML models.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    st.write("---")
    st.info("🔑 Google Fact Check API is enabled")
    st.caption("API key is configured in the code")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Select Text Column", df.columns)
    with col2:
        target_col = st.selectbox("Select Target Column", df.columns)

    phase = st.selectbox(
        "🔎 Choose NLP Phase",
        ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
    )

    if st.button("🚀 Run Model Comparison"):
        X = df[text_col].astype(str)
        y = df[target_col]

        # Feature extraction (ORIGINAL CODE - UNCHANGED)
        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_preprocess)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Semantic":
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                      columns=["polarity", "subjectivity"])

        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        else:  # Pragmatic
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                      columns=pragmatic_words)

        # Evaluate models (MODIFIED to return additional data)
        results, trained_models, test_indices = evaluate_models(X_features, y)
        
        # Display results (ORIGINAL CODE - UNCHANGED)
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df[results_df["Accuracy"].apply(lambda x: isinstance(x, (int,float)))]
        results_df = results_df.sort_values(by="Accuracy", ascending=False)

        st.subheader("🏆 Model Accuracy - User-Provided Labels")
        st.table(results_df)

        # Plot (ORIGINAL CODE - UNCHANGED)
        fig1 = plt.figure(figsize=(6, 4))
        plt.bar(results_df["Model"], results_df["Accuracy"], color="#4CAF50", alpha=0.8)
        plt.ylabel("Accuracy (%)")
        plt.title(f"Performance on {phase} (User-Provided Labels)")
        for i, v in enumerate(results_df["Accuracy"]):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
        
        # ============================
        # NEW: Google Fact Check Section
        # ============================
        st.write("---")
        st.subheader("🔍 Google Fact Check Comparison")
        
        # Use hardcoded API key
        if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
            # Show dataset info
            st.info(f"📊 Dataset: {len(y)} total statements | Test set: {len(test_indices)} statements (20%)")
            st.info(f"🔄 Now checking only the {len(test_indices)} test statements with Google Fact Check API...")
            
            with st.spinner("Fetching fact check data from Google API..."):
                progress_bar = st.progress(0)
                google_labels = get_google_fact_check_labels(X.tolist(), test_indices, GOOGLE_API_KEY, progress_bar)
                
                # Count how many Google labels we got
                available_count = sum(1 for i in test_indices if google_labels[i] is not None)
                
                st.info(f"📊 Google Fact Check returned data for {available_count} out of {len(test_indices)} test statements.")
                
                if available_count > 0:
                    # Evaluate models with Google labels
                    google_results = evaluate_with_google_labels(trained_models, google_labels, test_indices)
                    
                    # Create results dataframe - FIXED: Don't filter out valid results
                    google_results_df = pd.DataFrame(list(google_results.items()), columns=["Model", "Accuracy"])
                    
                    # Only filter out actual errors/N/A, not valid numbers
                    google_results_df_valid = google_results_df[
                        google_results_df["Accuracy"].apply(lambda x: isinstance(x, (int, float)))
                    ].copy()
                    
                    # Show results even if some models failed
                    if len(google_results_df_valid) > 0:
                        google_results_df_valid = google_results_df_valid.sort_values(by="Accuracy", ascending=False)
                        
                        st.success(f"✅ Successfully evaluated {len(google_results_df_valid)} models with {available_count} Google Fact Check labels!")
                        st.subheader("🎯 Model Accuracy - Google Fact Check Labels")
                        st.table(google_results_df_valid)
                        
                        # Plot for Google results
                        fig2 = plt.figure(figsize=(6, 4))
                        plt.bar(google_results_df_valid["Model"], google_results_df_valid["Accuracy"], color="#2196F3", alpha=0.8)
                        plt.ylabel("Accuracy (%)")
                        plt.title(f"Performance on {phase} (Google Fact Check Labels, N={available_count})")
                        for i, v in enumerate(google_results_df_valid["Accuracy"]):
                            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                        
                        # Comparison
                        st.subheader("📈 Comparison")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Best Model (User Labels)",
                                results_df.iloc[0]["Model"],
                                f"{results_df.iloc[0]['Accuracy']:.1f}%"
                            )
                        with col2:
                            st.metric(
                                "Best Model (Google Labels)",
                                google_results_df_valid.iloc[0]["Model"],
                                f"{google_results_df_valid.iloc[0]['Accuracy']:.1f}%"
                            )
                        with col3:
                            st.metric(
                                "Data Coverage",
                                f"{available_count}/{len(test_indices)}",
                                f"{(available_count/len(test_indices)*100):.0f}%"
                            )
                        
                        # Show note about sample size
                        if available_count < 10:
                            st.warning(f"⚠️ Note: Only {available_count} statements had Google Fact Check data. Results may not be very reliable. Try using more common factual statements for better coverage.")
                        elif available_count < 20:
                            st.info(f"ℹ️ Note: Evaluation based on {available_count} statements with Google Fact Check data.")
                    else:
                        st.error("❌ All models failed to evaluate. This shouldn't happen - please check your data.")
                else:
                    st.warning("⚠️ Google Fact Check API did not return data for any of your test statements. Try with more common/well-known factual claims.")
                    st.info("💡 Tip: Google Fact Check works best with mainstream news claims and widely-known factual statements.")
        else:
            st.warning("⚠️ Google Fact Check API key is not configured")
            st.info("💡 To enable Google Fact Check: Add your API key to the GOOGLE_API_KEY variable in the code (line ~40)")
            with st.expander("ℹ️ Why use Google Fact Check?"):
                st.markdown("""
                - **Validate your labels** against an external authoritative source
                - **Identify potential errors** in your manual labeling
                - **Add rigor** to your analysis by comparing with independent verification
                - **Demonstrate** real-world API integration skills
                """)

else:
    st.info("⬅️ Please upload a CSV file to start.")
    
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        ### Step 1: Upload Your Data
        - CSV file with:
          - A column containing text statements
          - A column with binary labels (0/1 or True/False)
        
        ### Step 2: Configure Settings
        1. Select your text and target columns
        2. Choose an NLP phase to analyze
        3. (Optional) Enter Google Fact Check API key for external validation
        
        ### Step 3: Run Analysis
        - Click "Run Model Comparison"
        - View model performance on user-provided labels
        - (If API key provided) Compare with Google Fact Check results
        
        ### Getting API Key (Optional)
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Enable "Fact Check Tools API"
        3. Create an API Key
        4. Paste it in the sidebar
        """)
