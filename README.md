# NLP Emotion Classification using Bag-of-Words & TF-IDF

A Machine Learning and Natural Language Processing project that classifies textual input into six distinct human emotions (**Joy, Sadness, Anger, Fear, Love, Surprise**). Includes exploratory analysis via Jupyter Notebook, model serialization (`.pkl`), and an interactive **Streamlit** web application.

---

## 📁 Repository Structure

```text
NLP-Project/
├── .gitignore                          # Comprehensive ignore rules for Python, venv, Jupyter, and Streamlit
├── requirements.txt                    # Project dependencies
├── train.txt                           # Labeled training dataset (16,000 text-emotion pairs)
├── NLP_using_Bag_of_Words_TF-IDF.ipynb # Notebook exploring BoW, TF-IDF, Naive Bayes & Logistic Regression
├── app.py                              # Interactive Streamlit Frontend Application
└── README.md                           # Documentation and setup guide
```

---

## 🛠️ Step 1: Virtual Environment Setup & Jupyter Kernel

To isolate project dependencies and avoid touching the global Python environment, a virtual environment (`venv`) is used.

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment named venv
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

*(If PowerShell execution policy blocks scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

### 2. Install Project Dependencies in `venv`

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Register the `venv` as a Jupyter Notebook Kernel

To use this virtual environment inside your Jupyter Notebook or VS Code Notebook:

```bash
python -m ipykernel install --user --name nlp-project-kernel --display-name "Python (NLP Project venv)"
```

In Jupyter Notebook or VS Code, select the kernel: **`Python (NLP Project venv)`**.

---

## 🧪 Step 2: How to Create and Export the `.pkl` File in Notebook / Google Colab

Run these cells in your **Google Colab** or **Jupyter Notebook** (`NLP_using_Bag_of_Words_TF-IDF.ipynb`).

### Cell A: Load Data & Preprocessing

```python
import string
import pickle
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

# Map emotions to integers
unique_emotions = df['emotion'].unique()
emotion_to_id = {emo: i for i, emo in enumerate(unique_emotions)}
id_to_emotion = {i: emo for emo, i in emotion_to_id.items()}
df['label'] = df['emotion'].map(emotion_to_id)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([c for c in text if not c.isdigit()])
    text = ''.join([c for c in text if c.isascii()])
    words = word_tokenize(text)
    return ' '.join([w for w in words if w not in stop_words])

df['cleaned_text'] = df['text'].apply(clean_text)
```

### Cell B: Train TF-IDF & Logistic Regression

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.20, random_state=42
)

# 1. Fit TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 2. Fit Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

print("Test Accuracy:", round(logistic_model.score(X_test_tfidf, y_test) * 100, 2), "%")
```

### Cell C: Save the Model and Vectorizer to `.pkl`

You can save them in a `models/` directory or in the project root:

```python
# Create models directory
os.makedirs('models', exist_ok=True)

# Option 1: Save separate files
with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('models/id_to_emotion.pkl', 'wb') as f:
    pickle.dump(id_to_emotion, f)

# Option 2: Save as a single combined bundle
bundle = {
    'model': logistic_model,
    'vectorizer': tfidf_vectorizer,
    'id_to_emotion': id_to_emotion
}
with open('models/model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("Pickle files saved successfully!")
```

### Cell D: (Google Colab only) Download `.pkl` to your Local Machine

```python
from google.colab import files
files.download('models/model.pkl')
# or
files.download('models/logistic_model.pkl')
files.download('models/tfidf_vectorizer.pkl')
```

Place the downloaded `.pkl` file into the `d:\NLP-Project\models\` folder (or directly in `d:\NLP-Project\`).

---

## 🚀 Step 3: Run the Streamlit Frontend

With your virtual environment activated:

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

Or when `venv` is activated:
```powershell
streamlit run app.py
```

### Features of the Frontend:
1. **Single Text Emotion Predictor:** Real-time sentiment classification with probability breakdowns across all 6 emotions and confidence scoring.
2. **Interactive Presets:** 1-click test sentences covering Joy, Sadness, Anger, Fear, Love, and Surprise.
3. **Batch File Analysis:** Upload CSV or TXT files to process hundreds of texts simultaneously and download results with a click.
4. **Preprocessing Pipeline Inspector:** Visualizes raw text transformations (lowercasing, punctuation stripping, emoji filtering, stopword elimination).
5. **Interactive Pickle Upload & Quick Local Generation:** Upload `.pkl` files through the browser or generate locally in 1 click.

---

## 🔗 Step 4: Git & GitHub Setup

This repository is initialized with local git identity:
- **Username:** `utsav210`
- **Email:** `utsavgandhi273@gmail.com`

To link this repository to a remote repository on GitHub:

```bash
# 1. Add your GitHub remote repository URL
git remote add origin https://github.com/utsav210/<YOUR-REPO-NAME>.git

# 2. Push to GitHub
git branch -M main
git push -u origin main
```
