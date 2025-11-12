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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
# Google Fact Check API Integration
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
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "claims" in data and len(data["claims"]) > 0:
                # Get the first claim's rating
                first_claim = data["claims"][0]
                if "claimReview" in first_claim and len(first_claim["claimReview"]) > 0:
                    rating = first_claim["claimReview"][0].get("textualRating", "").lower()
                    
                    # Map ratings to binary (you can customize this mapping)
                    true_ratings = ["true", "mostly true", "correct", "accurate"]
                    false_ratings = ["false", "mostly false", "incorrect", "inaccurate", "pants on fire"]
                    
                    if any(tr in rating for tr in true_ratings):
                        return 1
                    elif any(fr in rating for fr in false_ratings):
                        return 0
        return None
    except Exception as e:
        st.warning(f"Error checking fact: {str(e)}")
        return None

def get_google_fact_check_labels(texts, api_key, progress_bar=None):
    """
    Get fact check labels for a list of texts using Google Fact Check API.
    """
    labels = []
    for idx, text in enumerate(texts):
        label = check_fact_with_google(text, api_key)
        labels.append(label)
        
        if progress_bar:
            progress_bar.progress((idx + 1) / len(texts))
        
        # Rate limiting: wait a bit between requests to avoid hitting API limits
        time.sleep(0.5)
    
    return labels

# ============================
# Feature Extractors
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
# Model Evaluation Functions
# ============================
def evaluate_models(X_features, y):
    """Evaluate models and return predictions along with metrics"""
    results = {}
    model_predictions = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate multiple metrics
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            
            results[name] = {
                "Accuracy": round(acc, 2),
                "Precision": round(prec, 2),
                "Recall": round(rec, 2),
                "F1-Score": round(f1, 2)
            }
            
            # Store predictions for later comparison with Google Fact Check
            model_predictions[name] = {
                "model": model,
                "test_indices": X_test.index if hasattr(X_test, 'index') else list(range(len(X_test)))
            }
            
        except Exception as e:
            results[name] = {
                "Accuracy": f"Error: {str(e)}",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1-Score": "N/A"
            }
            model_predictions[name] = None
    
    return results, model_predictions, X_test, y_test

def evaluate_against_google_factcheck(model_predictions, X_features, google_labels, original_indices):
    """
    Evaluate model predictions against Google Fact Check labels
    """
    results = {}
    
    for model_name, pred_data in model_predictions.items():
        if pred_data is None:
            results[model_name] = {
                "Accuracy": "N/A",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1-Score": "N/A",
                "Available Data": 0
            }
            continue
        
        model = pred_data["model"]
        test_indices = pred_data["test_indices"]
        
        # Get Google labels for test set
        google_test_labels = [google_labels[i] for i in test_indices if google_labels[i] is not None]
        
        if len(google_test_labels) == 0:
            results[model_name] = {
                "Accuracy": "No data",
                "Precision": "No data",
                "Recall": "No data",
                "F1-Score": "No data",
                "Available Data": 0
            }
            continue
        
        # Get corresponding predictions
        valid_test_indices = [i for i, idx in enumerate(test_indices) if google_labels[idx] is not None]
        
        if hasattr(X_features, 'iloc'):
            X_test_valid = X_features.iloc[test_indices].iloc[valid_test_indices]
        else:
            X_test_valid = X_features[test_indices][valid_test_indices]
        
        y_pred = model.predict(X_test_valid)
        
        # Calculate metrics
        acc = accuracy_score(google_test_labels, y_pred) * 100
        prec = precision_score(google_test_labels, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(google_test_labels, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(google_test_labels, y_pred, average='weighted', zero_division=0) * 100
        
        results[model_name] = {
            "Accuracy": round(acc, 2),
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1-Score": round(f1, 2),
            "Available Data": len(google_test_labels)
        }
    
    return results

# ============================
# Visualization Functions
# ============================
def plot_comparison(results_df, title):
    """Create a bar plot for model comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(results_df))
    width = 0.2
    
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            values = results_df[metric].apply(lambda x: x if isinstance(x, (int, float)) else 0)
            ax.bar([p + width * i for p in x], values, width, label=metric, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score (%)')
    ax.set_title(title)
    ax.set_xticks([p + width * 1.5 for p in x])
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase-wise Analysis with Google Fact Check", layout="wide")
st.title("🧠 NLP Phase-wise Analysis with Google Fact Check")
st.markdown(
    "<p style='color:gray;'>Upload a dataset, choose an NLP phase, and compare multiple ML models against both user-provided labels and Google Fact Check API.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    st.write("---")
    
    st.header("🔑 Google Fact Check API")
    st.markdown("Get your API key from [Google Cloud Console](https://console.cloud.google.com/apis/library/factchecktools.googleapis.com)")
    api_key = st.text_input("Enter API Key", type="password")
    
    if api_key:
        st.success("✅ API Key provided")
    else:
        st.info("💡 Optional: Add API key for fact checking")
    
    st.write("---")

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
        with st.spinner("Processing data and training models..."):
            X = df[text_col].astype(str)
            y = df[target_col]

            # Feature extraction based on phase
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

            # Evaluate models with user-provided labels
            results, model_predictions, X_test, y_test = evaluate_models(X_features, y)
            
            # Create results dataframe for user-provided labels
            results_list = []
            for model_name, metrics in results.items():
                if isinstance(metrics, dict):
                    row = {"Model": model_name}
                    row.update(metrics)
                    results_list.append(row)
            
            results_df = pd.DataFrame(results_list)
            results_df = results_df[results_df["Accuracy"].apply(lambda x: isinstance(x, (int, float)))]
            results_df = results_df.sort_values(by="Accuracy", ascending=False)

            # Display results for user-provided labels
            st.subheader("🏆 Model Performance - User-Provided Labels")
            st.table(results_df)

            # Plot for user-provided labels
            fig1 = plot_comparison(results_df, f"Performance on {phase} (User-Provided Labels)")
            st.pyplot(fig1)

            # Google Fact Check Section
            st.write("---")
            st.subheader("🔍 Google Fact Check Comparison")
            
            if api_key and api_key.strip() != "":
                with st.spinner("Fetching fact check data from Google API... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    google_labels = get_google_fact_check_labels(X.tolist(), api_key, progress_bar)
                    
                    # Count available labels
                    available_count = sum(1 for label in google_labels if label is not None)
                    
                    st.info(f"📊 Google Fact Check API returned data for {available_count} out of {len(google_labels)} statements.")
                    
                    if available_count > 0:
                        # Evaluate models against Google Fact Check labels
                        google_results = evaluate_against_google_factcheck(
                            model_predictions, X_features, google_labels, list(range(len(X)))
                        )
                        
                        # Create results dataframe for Google Fact Check
                        google_results_list = []
                        for model_name, metrics in google_results.items():
                            row = {"Model": model_name}
                            row.update(metrics)
                            google_results_list.append(row)
                        
                        google_results_df = pd.DataFrame(google_results_list)
                        
                        # Filter out models with no data
                        google_results_df_filtered = google_results_df[
                            google_results_df["Accuracy"].apply(lambda x: isinstance(x, (int, float)))
                        ]
                        
                        if not google_results_df_filtered.empty:
                            google_results_df_filtered = google_results_df_filtered.sort_values(by="Accuracy", ascending=False)
                            
                            st.subheader("🎯 Model Performance - Google Fact Check Labels")
                            st.table(google_results_df_filtered)
                            
                            # Plot for Google Fact Check
                            fig2 = plot_comparison(google_results_df_filtered, f"Performance on {phase} (Google Fact Check Labels)")
                            st.pyplot(fig2)
                            
                            # Comparison insights
                            st.subheader("📈 Comparison Insights")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Best Model (User Labels)",
                                    results_df.iloc[0]["Model"],
                                    f"{results_df.iloc[0]['Accuracy']:.2f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Best Model (Google Labels)",
                                    google_results_df_filtered.iloc[0]["Model"],
                                    f"{google_results_df_filtered.iloc[0]['Accuracy']:.2f}%"
                                )
                        else:
                            st.warning("⚠️ Not enough Google Fact Check data available to evaluate models.")
                    else:
                        st.warning("⚠️ Google Fact Check API did not return any data for the provided statements. This could mean the statements are not commonly fact-checked or are too specific.")
            else:
                st.info("🔑 Please provide a Google Fact Check API key in the sidebar to enable this feature.")

else:
    st.info("⬅️ Please upload a CSV file to start.")
    
    # Instructions section
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        ### Step 1: Get Google Fact Check API Key
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. Enable the "Fact Check Tools API"
        4. Create credentials (API Key)
        5. Copy the API key and paste it in the sidebar
        
        ### Step 2: Upload Your Data
        - Your CSV should have:
          - A column with text statements
          - A column with binary labels (0/1 or True/False)
        
        ### Step 3: Configure and Run
        1. Select your text and target columns
        2. Choose an NLP phase to analyze
        3. Click "Run Model Comparison"
        
        ### Step 4: Interpret Results
        - **User-Provided Labels**: Shows how models perform against your labels
        - **Google Fact Check Labels**: Shows how models perform against Google's fact-checking data
        - Compare both to understand model reliability
        """)
