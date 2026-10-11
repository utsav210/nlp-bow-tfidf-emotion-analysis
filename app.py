"""
Streamlit Web Application for NLP Emotion Detection
Uses Bag-of-Words & TF-IDF with Machine Learning models.
Supports loading .pkl artifacts generated from Jupyter Notebook / Google Colab.
"""

import os
import re
import string
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np

# Optional Plotly for charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Fallback English stopwords list
DEFAULT_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
}

# Emotion visual metadata
EMOTION_META = {
    "joy": {"emoji": "😊", "color": "#10B981", "badge": "Joy", "desc": "Happiness, delight, satisfaction"},
    "sadness": {"emoji": "😢", "color": "#3B82F6", "badge": "Sadness", "desc": "Grief, sorrow, despair"},
    "anger": {"emoji": "😡", "color": "#EF4444", "badge": "Anger", "desc": "Frustration, fury, resentment"},
    "fear": {"emoji": "😨", "color": "#8B5CF6", "badge": "Fear", "desc": "Anxiety, apprehension, dread"},
    "love": {"emoji": "❤️", "color": "#EC4899", "badge": "Love", "desc": "Affection, warmth, romance"},
    "surprise": {"emoji": "😲", "color": "#F59E0B", "badge": "Surprise", "desc": "Astonishment, amazement, wonder"}
}

DEFAULT_ID_TO_EMOTION = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}


# -------------------------------------------------------------
# Text Preprocessing Pipeline
# -------------------------------------------------------------
@st.cache_resource
def load_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words('english'))
        except Exception:
            nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english'))
    except Exception:
        return DEFAULT_STOPWORDS


def clean_text_pipeline(text: str, stop_words: set):
    """
    Step-by-step cleaning matching the notebook logic:
    1. Lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove emojis (keep ASCII)
    5. Remove stopwords
    Returns cleaned string and intermediate steps for inspection.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    steps = {}
    steps["original"] = text

    # Step 1: Lowercase
    s1 = text.lower()
    steps["lowercase"] = s1

    # Step 2: Remove punctuation
    s2 = s1.translate(str.maketrans('', '', string.punctuation))
    steps["no_punctuation"] = s2

    # Step 3: Remove numbers
    s3 = "".join([c for c in s2 if not c.isdigit()])
    steps["no_numbers"] = s3

    # Step 4: Remove emojis / non-ascii
    s4 = "".join([c for c in s3 if c.isascii()])
    steps["no_emojis"] = s4

    # Step 5: Stopwords removal
    tokens = s4.split()
    kept_tokens = [w for w in tokens if w not in stop_words]
    removed_stopwords = [w for w in tokens if w in stop_words]
    cleaned = " ".join(kept_tokens)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    steps["stopwords_removed"] = removed_stopwords
    steps["cleaned"] = cleaned
    steps["tokens"] = kept_tokens

    return cleaned, steps


# -------------------------------------------------------------
# Pickle Model Loader & Auto-Discovery
# -------------------------------------------------------------
def find_and_load_artifacts():
    """
    Discovers .pkl files in the current folder or 'models/' directory.
    Supports single bundled files or separate vectorizer & model files.
    """
    search_dirs = [".", "models"]
    model = None
    vectorizer = None
    id_to_emotion = DEFAULT_ID_TO_EMOTION.copy()
    loaded_source = None

    # Check for uploaded file in session state
    if "custom_model_bytes" in st.session_state and st.session_state.custom_model_bytes:
        try:
            loaded_obj = pickle.loads(st.session_state.custom_model_bytes)
            if isinstance(loaded_obj, dict):
                model = loaded_obj.get("model") or loaded_obj.get("classifier")
                vectorizer = loaded_obj.get("vectorizer")
                if "id_to_emotion" in loaded_obj:
                    id_to_emotion = loaded_obj["id_to_emotion"]
            else:
                model = loaded_obj
            loaded_source = "Uploaded Model File"
        except Exception as e:
            st.error(f"Error reading uploaded pickle: {e}")

    # Check for bundle file: model.pkl or emotion_model.pkl
    if model is None or vectorizer is None:
        for d in search_dirs:
            for bundle_name in ["model.pkl", "emotion_model.pkl", "pipeline.pkl"]:
                p = os.path.join(d, bundle_name)
                if os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            obj = pickle.load(f)
                        if isinstance(obj, dict):
                            model = obj.get("model") or obj.get("classifier") or model
                            vectorizer = obj.get("vectorizer") or vectorizer
                            if "id_to_emotion" in obj:
                                id_to_emotion = obj["id_to_emotion"]
                            loaded_source = f"{p} (Bundle)"
                            break
                        elif hasattr(obj, "predict"):
                            # Might be pipeline or model
                            model = obj
                            loaded_source = f"{p} (Direct Model)"
                    except Exception:
                        pass
            if model and vectorizer:
                break

    # Look for individual model and vectorizer files
    if model is None or vectorizer is None:
        for d in search_dirs:
            # Model files
            for m_name in ["logistic_model.pkl", "nb_model.pkl", "model.pkl", "classifier.pkl"]:
                p = os.path.join(d, m_name)
                if model is None and os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            model = pickle.load(f)
                            loaded_source = (loaded_source + f" + {p}") if loaded_source else p
                    except Exception:
                        pass

            # Vectorizer files
            for v_name in ["tfidf_vectorizer.pkl", "vectorizer.pkl", "bow_vectorizer.pkl"]:
                p = os.path.join(d, v_name)
                if vectorizer is None and os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            vectorizer = pickle.load(f)
                    except Exception:
                        pass

            # Emotion mapping file
            for map_name in ["emotion_map.pkl", "id_to_emotion.pkl", "label_mapping.pkl", "emotion_mapping.pkl"]:
                p = os.path.join(d, map_name)
                if os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            id_to_emotion = pickle.load(f)
                    except Exception:
                        pass

    # Normalize integer keys in mapping
    if id_to_emotion:
        id_to_emotion = {int(k): str(v).lower() for k, v in id_to_emotion.items()}

    return model, vectorizer, id_to_emotion, loaded_source


# -------------------------------------------------------------
# Optional Local Training Fallback (if user runs locally)
# -------------------------------------------------------------
def train_models_locally():
    """Trains TF-IDF + Logistic Regression from train.txt and saves .pkl files."""
    if not os.path.exists("train.txt"):
        st.error("Cannot train: 'train.txt' not found in workspace.")
        return False

    with st.spinner("Reading train.txt and preprocessing data..."):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])
        stop_words = load_stopwords()

        unique_emotions = df["emotion"].unique()
        emotion_to_id = {emo: i for i, emo in enumerate(unique_emotions)}
        id_to_emotion = {i: emo for emo, i in emotion_to_id.items()}
        df["label"] = df["emotion"].map(emotion_to_id)

        df["cleaned"] = df["text"].apply(lambda t: clean_text_pipeline(t, stop_words)[0])

        vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df["cleaned"])
        y = df["label"].values

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        os.makedirs("models", exist_ok=True)
        with open("models/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open("models/logistic_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("models/id_to_emotion.pkl", "wb") as f:
            pickle.dump(id_to_emotion, f)

        # Also save unified bundle for convenience
        with open("models/model.pkl", "wb") as f:
            pickle.dump({"model": model, "vectorizer": vectorizer, "id_to_emotion": id_to_emotion}, f)

    return True


# -------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------
def predict_emotion(text: str, model, vectorizer, id_to_emotion, stop_words):
    cleaned, steps = clean_text_pipeline(text, stop_words)
    if not cleaned:
        return None, None, steps

    # Check if model is an end-to-end Pipeline
    if vectorizer is None and hasattr(model, "predict"):
        pred_label = model.predict([cleaned])[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
        else:
            probs = None
    else:
        vec = vectorizer.transform([cleaned])
        pred_label = model.predict(vec)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
        else:
            probs = None

    emotion_name = id_to_emotion.get(pred_label, f"Class {pred_label}")

    prob_dict = {}
    if probs is not None:
        # Match probabilities to emotion labels
        for idx, p in enumerate(probs):
            emo = id_to_emotion.get(idx, f"Class {idx}")
            prob_dict[emo] = float(p)

    return emotion_name, prob_dict, steps


# -------------------------------------------------------------
# Streamlit Application Setup & UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="EmotionAI - NLP Emotion Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .stTextArea textarea {
        font-size: 1.05rem;
        border-radius: 10px;
    }
    .code-box {
        background-color: #1e1e2e;
        color: #f8f8f2;
        padding: 14px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("EmotionAI Studio")
st.sidebar.caption("Bag-of-Words & TF-IDF NLP Project")

stop_words = load_stopwords()
model, vectorizer, id_to_emotion, loaded_source = find_and_load_artifacts()

# Sidebar Status
st.sidebar.markdown("---")
st.sidebar.subheader(" Model Status")
if model is not None and (vectorizer is not None or hasattr(model, "predict")):
    st.sidebar.success(" Model Loaded & Ready")
    st.sidebar.markdown(f"**Source:** `{loaded_source}`")
    st.sidebar.markdown(f"**Recognized Classes:** {len(id_to_emotion)}")
else:
    st.sidebar.warning(" No .pkl Model Found Yet")
    st.sidebar.info("Follow the instructions in the **'Notebook Pickle Guide'** tab or use the uploader below.")

# Sidebar File Uploader for Custom Pickle
st.sidebar.markdown("---")
st.sidebar.subheader(" Upload .pkl Artifact")
uploaded_pkl = st.sidebar.file_uploader("Upload model.pkl or vectorizer.pkl", type=["pkl", "joblib"])
if uploaded_pkl is not None:
    st.session_state.custom_model_bytes = uploaded_pkl.read()
    st.sidebar.success("Uploaded file applied! Reloading...")
    st.rerun()

# Sidebar Quick Local Trainer button
if st.sidebar.button("⚙️ Train & Generate .pkl Locally", help="Trains TF-IDF + Logistic Regression on train.txt and saves models/*.pkl"):
    success = train_models_locally()
    if success:
        st.sidebar.success("Training complete! .pkl files saved in 'models/'.")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("👤 **Author:** utsav210")
st.sidebar.markdown("📧 **Email:** utsavgandhi273@gmail.com")

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.title("🧠 NLP Emotion Detection Platform")
st.markdown(
    "Detect and analyze emotional sentiment in natural text using **Bag-of-Words**, **TF-IDF Vectorization**, "
    "and **Logistic Regression** trained on 16,000+ annotated records."
)

# Main Navigation Tabs
tab_predict, tab_batch, tab_pipeline, tab_guide, tab_dataset = st.tabs([
    "🎯 Single Prediction",
    "📂 Batch File Analysis",
    "🔬 Preprocessing Pipeline",
    "📘 Notebook Pickle Guide",
    "📊 Dataset & Metrics"
])


# -------------------------------------------------------------
# TAB 1: Single Prediction
# -------------------------------------------------------------
with tab_predict:
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("Enter Text to Analyze")

        # Quick preset examples
        st.markdown("**Quick Preset Examples:**")
        preset_cols = st.columns(3)
        sample_prompts = [
            ("😊 Joy", "I just got promoted today and my team celebrated with me!"),
            ("😢 Sadness", "I feel so lonely and hopeless sitting here in the dark."),
            ("😡 Anger", "This is utterly unacceptable and I am completely furious about it!"),
            ("😨 Fear", "I am feeling terrified about the medical report tomorrow."),
            ("❤️ Love", "Every single moment spent with you makes my heart so warm."),
            ("😲 Surprise", "I was completely blown away when everyone shouted happy birthday!")
        ]

        preset_text = None
        for i, (label, text_val) in enumerate(sample_prompts):
            with preset_cols[i % 3]:
                if st.button(label, key=f"btn_preset_{i}", use_container_width=True):
                    preset_text = text_val

        default_val = preset_text if preset_text else "I am so grateful and overjoyed by all the support I received today!"
        user_text = st.text_area(
            "Type or paste a sentence/tweet/review:",
            value=default_val,
            height=130
        )

        col_w, col_c = st.columns([1, 1])
        with col_w:
            word_count = len(user_text.split()) if user_text else 0
            st.caption(f"📝 Words: {word_count} | Characters: {len(user_text)}")

        analyze_clicked = st.button("🚀 Analyze Emotion", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction & Confidence")

        if model is None:
            st.warning("⚠️ No `.pkl` model found yet. Please export it from your notebook or use the **Notebook Pickle Guide** tab!")
        else:
            if user_text.strip():
                pred_emotion, prob_dict, steps = predict_emotion(
                    user_text, model, vectorizer, id_to_emotion, stop_words
                )

                if pred_emotion:
                    meta = EMOTION_META.get(pred_emotion.lower(), {
                        "emoji": "🎭", "color": "#6366F1", "badge": pred_emotion.title(), "desc": ""
                    })

                    top_prob = prob_dict.get(pred_emotion, 0.0) if prob_dict else 1.0

                    # Hero Result Card
                    st.markdown(f"""
                    <div style="background-color: {meta['color']}18; border: 2px solid {meta['color']}; border-radius: 14px; padding: 22px; text-align: center; margin-bottom: 20px;">
                        <span style="font-size: 3.2rem;">{meta['emoji']}</span>
                        <h2 style="margin: 6px 0; color: {meta['color']};">{meta['badge']}</h2>
                        <p style="margin: 0; opacity: 0.85; font-size: 1.05rem;">{meta['desc']}</p>
                        <div style="margin-top: 10px;">
                            <span class="badge-pill" style="background-color: {meta['color']}; color: white;">
                                Confidence: {top_prob * 100:.2f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probabilities Breakdown
                    if prob_dict and HAS_PLOTLY:
                        st.markdown("**Emotion Probability Distribution:**")
                        df_probs = pd.DataFrame([
                            {
                                "Emotion": emo.capitalize(),
                                "Probability (%)": round(p * 100, 2),
                                "Color": EMOTION_META.get(emo.lower(), {}).get("color", "#6366F1")
                            }
                            for emo, p in prob_dict.items()
                        ]).sort_values("Probability (%)", ascending=True)

                        fig = px.bar(
                            df_probs,
                            x="Probability (%)",
                            y="Emotion",
                            orientation="h",
                            color="Emotion",
                            color_discrete_map={row["Emotion"]: row["Color"] for _, row in df_probs.iterrows()},
                            text="Probability (%)"
                        )
                        fig.update_layout(
                            showlegend=False,
                            margin=dict(l=0, r=0, t=10, b=10),
                            height=240,
                            xaxis_title="Confidence (%)",
                            yaxis_title=""
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    elif prob_dict:
                        # Fallback Streamlit bar chart
                        st.bar_chart(pd.Series(prob_dict) * 100)
                else:
                    st.info("Input was empty after cleaning. Please enter meaningful text.")
            else:
                st.info("Enter text in the left panel to see predictions.")


# -------------------------------------------------------------
# TAB 2: Batch Analysis (CSV / TXT)
# -------------------------------------------------------------
with tab_batch:
    st.subheader("Batch File Emotion Classifier")
    st.markdown("Upload a `.csv` or `.txt` file with text lines to classify sentiments in bulk.")

    col_u, col_opt = st.columns([2, 1])
    with col_u:
        batch_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"], key="batch_uploader")

    if batch_file is not None:
        try:
            if batch_file.name.endswith(".csv"):
                batch_df = pd.read_csv(batch_file)
                text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
                if not text_cols:
                    st.error("No text column found in CSV.")
                    st.stop()
                selected_col = st.selectbox("Select text column to analyze:", text_cols)
                texts = batch_df[selected_col].astype(str).tolist()
            else:
                raw_content = batch_file.read().decode("utf-8", errors="ignore")
                lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
                # If separated by ';', take first part (like train.txt)
                texts = [line.split(";")[0] if ";" in line else line for line in lines]
                batch_df = pd.DataFrame({"text": texts})
                selected_col = "text"

            st.write(f"Loaded **{len(texts)}** records.")

            if model is None:
                st.warning("Please load or provide the `.pkl` model first!")
            else:
                if st.button("⚡ Run Batch Prediction", type="primary"):
                    with st.spinner(f"Analyzing {len(texts)} records..."):
                        preds = []
                        confidences = []
                        for t in texts:
                            emo, probs, _ = predict_emotion(t, model, vectorizer, id_to_emotion, stop_words)
                            preds.append(emo if emo else "Unknown")
                            top_p = probs.get(emo, 0.0) if (probs and emo) else 0.0
                            confidences.append(round(top_p * 100, 2))

                        batch_df["Predicted_Emotion"] = preds
                        batch_df["Confidence (%)"] = confidences

                        st.success(f"Successfully processed {len(texts)} items!")

                        # Metrics overview
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Processed", len(batch_df))
                        top_emotion = batch_df["Predicted_Emotion"].mode()[0]
                        m2.metric("Dominant Emotion", f"{EMOTION_META.get(top_emotion, {}).get('emoji', '')} {top_emotion.title()}")
                        m3.metric("Average Confidence", f"{batch_df['Confidence (%)'].mean():.2f}%")

                        # Chart
                        if HAS_PLOTLY:
                            counts = batch_df["Predicted_Emotion"].value_counts().reset_index()
                            counts.columns = ["Emotion", "Count"]
                            fig_batch = px.pie(
                                counts,
                                names="Emotion",
                                values="Count",
                                title="Batch Emotion Distribution",
                                color="Emotion",
                                color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()}
                            )
                            st.plotly_chart(fig_batch, use_container_width=True)

                        st.dataframe(batch_df, use_container_width=True)

                        csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_bytes,
                            file_name="emotion_predictions.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing file: {e}")


# -------------------------------------------------------------
# TAB 3: Preprocessing Pipeline
# -------------------------------------------------------------
with tab_pipeline:
    st.subheader("NLP Preprocessing Pipeline Inspection")
    st.markdown(
        "Explore how raw unstructured human sentences are cleaned step-by-step before being transformed into numbers."
    )

    test_input = st.text_input(
        "Test text for pipeline inspection:",
        value="I didn't feel humiliated! I can go from feeling hopeless to damned hopeful... #Life 2026 😊"
    )

    cleaned_res, steps_res = clean_text_pipeline(test_input, stop_words)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Raw Input:**")
        st.code(steps_res["original"])

        st.markdown("**2. Lowercased:**")
        st.code(steps_res["lowercase"])

        st.markdown("**3. Punctuation Removed:**")
        st.code(steps_res["no_punctuation"])

    with c2:
        st.markdown("**4. Digits Removed:**")
        st.code(steps_res["no_numbers"])

        st.markdown("**5. Emojis Removed (ASCII only):**")
        st.code(steps_res["no_emojis"])

        st.markdown("**6. Final Cleaned (Stopwords Removed):**")
        st.success(steps_res["cleaned"] if steps_res["cleaned"] else "(Empty string)")

    st.markdown("---")
    st.markdown("**Removed Stopwords in this sentence:**")
    if steps_res["stopwords_removed"]:
        st.write(", ".join([f"`{w}`" for w in set(steps_res["stopwords_removed"])]))
    else:
        st.write("None")

    if vectorizer is not None and steps_res["cleaned"]:
        st.markdown("---")
        st.markdown("**TF-IDF Vector Representation Sample:**")
        vec = vectorizer.transform([steps_res["cleaned"]])
        non_zero_indices = vec.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        tfidf_pairs = [{"Feature / N-gram": feature_names[i], "TF-IDF Weight": round(vec[0, i], 4)} for i in non_zero_indices]
        if tfidf_pairs:
            st.table(pd.DataFrame(tfidf_pairs).sort_values("TF-IDF Weight", ascending=False))


# -------------------------------------------------------------
# TAB 4: Notebook Pickle Guide
# -------------------------------------------------------------
with tab_guide:
    st.subheader("📘 How to Create and Export the .pkl File in your Notebook")
    st.markdown("""
    Follow these easy steps in your **Jupyter Notebook** or **Google Colab** to train the model and save the `.pkl` files.
    Once created, download the `.pkl` files and place them in the project folder (or in a `models/` directory),
    or upload them via the left sidebar.
    """)

    st.markdown("### Step 1: Install & Import Dependencies")
    st.code("""
import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
""", language="python")

    st.markdown("### Step 2: Load Data & Preprocess")
    st.code("""
# Load dataset
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

# Encode emotions into integers
unique_emotions = df['emotion'].unique()
emotion_to_id = {emo: i for i, emo in enumerate(unique_emotions)}
id_to_emotion = {i: emo for emo, i in emotion_to_id.items()}
df['emotion'] = df['emotion'].map(emotion_to_id)

# Cleaning functions
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([c for c in text if not c.isdigit()])
    text = ''.join([c for c in text if c.isascii()])
    words = word_tokenize(text)
    return ' '.join([w for w in words if w not in stop_words])

df['cleaned_text'] = df['text'].apply(clean_text)
""", language="python")

    st.markdown("### Step 3: Train TF-IDF & Logistic Regression")
    st.code("""
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['emotion'], test_size=0.20, random_state=42
)

# 1. Fit TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 2. Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

print("Test Accuracy:", logistic_model.score(X_test_tfidf, y_test))
""", language="python")

    st.markdown("### Step 4: Export to .pkl Files")
    st.markdown("Run this in your notebook to save the exact files recognized by this Streamlit app:")
    st.code("""
# Option A: Save separate files
import os
os.makedirs('models', exist_ok=True)

with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('models/id_to_emotion.pkl', 'wb') as f:
    pickle.dump(id_to_emotion, f)

# Option B: Or save a single bundled model.pkl file
bundle = {
    'model': logistic_model,
    'vectorizer': tfidf_vectorizer,
    'id_to_emotion': id_to_emotion
}
with open('models/model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print(" Pickles exported successfully!")
""", language="python")

    st.markdown("### Step 5: If in Google Colab, Download to your Local Machine")
    st.code("""
from google.colab import files
files.download('models/model.pkl')
# or files.download('models/logistic_model.pkl')
# or files.download('models/tfidf_vectorizer.pkl')
""", language="python")
    st.info("Place the downloaded `.pkl` files in `d:/NLP-Project/models/` or upload them using the left sidebar in this app!")


# -------------------------------------------------------------
# TAB 5: Dataset & Metrics
# -------------------------------------------------------------
with tab_dataset:
    st.subheader("Dataset & Model Evaluation")

    if os.path.exists("train.txt"):
        @st.cache_data
        def load_raw_dataset():
            return pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

        df_train = load_raw_dataset()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", f"{len(df_train):,}")
        c2.metric("Emotion Categories", df_train["emotion"].nunique())
        c3.metric("Vocabulary Size", "~15,000 features")

        st.markdown("### Emotion Class Distribution in `train.txt`")
        counts_df = df_train["emotion"].value_counts().reset_index()
        counts_df.columns = ["Emotion", "Count"]
        counts_df["Percentage (%)"] = (counts_df["Count"] / len(df_train) * 100).round(2)

        if HAS_PLOTLY:
            fig_dist = px.bar(
                counts_df,
                x="Emotion",
                y="Count",
                color="Emotion",
                color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()},
                text="Count"
            )
            fig_dist.update_traces(textposition='outside')
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.dataframe(counts_df, use_container_width=True)

        st.markdown("### Model Benchmark Comparison (from Notebook)")
        comparison_data = pd.DataFrame([
            {"Feature Extraction": "Bag-of-Words (CountVectorizer)", "Classifier": "Multinomial Naive Bayes", "Expected Accuracy": "~79.5%"},
            {"Feature Extraction": "TF-IDF Vectorizer", "Classifier": "Multinomial Naive Bayes", "Expected Accuracy": "~77.2%"},
            {"Feature Extraction": "TF-IDF Vectorizer (N-grams)", "Classifier": "Logistic Regression", "Expected Accuracy": "~88.6% (Best)"}
        ])
        st.table(comparison_data)
    else:
        st.warning("`train.txt` not found in current working directory.")
