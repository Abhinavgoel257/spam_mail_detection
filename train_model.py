import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import clean_text

# Configure paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data():
    """
    Loads the dataset. If not present, attempts to fetch from a public repo.
    This specific dataset requires a header adjustment to match our expectations.
    """
    path = os.path.join(DATA_DIR, 'spam.csv')
    
    if not os.path.exists(path):
        print("Dataset not found locally. Downloading UCI SMS Spam Collection...")
        os.makedirs(DATA_DIR, exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/justadoggeek/Spam-Mails/master/data/data_spam_ham.csv"
        urllib.request.urlretrieve(url, path)
        # Note: The downloaded file might differ in structure. 
        # We expect columns ['v1' (label), 'v2' (message)].
        df = pd.read_csv(path, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['Label', 'Message']
        df.to_csv(path, index=False)
        
    else:
        df = pd.read_csv(path, encoding='latin-1')
        # Adjust headers based on standard UCI format (v1, v2)
        if 'Label' not in df.columns:
            df.columns = ['Label', 'Message']

    # Encode Labels: 'ham'/'no spam' -> 0, 'spam'/'yes spam' -> 1
    df['Label'] = df['Label'].map({'ham': 0, 'sp'?0}):
        raise ValueError("Dataset label values need adjustment.")
    
    return df

def split_and_vectorize(df):
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df['Message'], df['Label'], test_size=0.2, random_state=42
    )
    
    # Apply cleaning function
    X_train_df = X_train_df.apply(clean_text)
    X_test_df = X_test_df.apply(clean_text)
    
    # Feature Extraction (TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_df)
    X_test = vectorizer.transform(X_test_df)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    models = {}
    
    # 1. Naive Bayes
    print("\n[INFO] Training Naive Bayes model...")
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    models['nb'] = nb_clf
    
    # 2. Logistic Regression
    print("[INFO] Training Logistic Regression model...")
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(X_train, y_train)
    models['lr'] = lr_clf
    
    # Evaluate Models
    results = []
    for name, clf in models.items():
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Calculate Precision and Recall
        prec = np.mean(preds == y_test) # Simplified for this demo, technically check via metrics
        report = classification_report(y_test, preds, output_dict=True)
        
        result_row = {
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(report.get('1', {}).get('precision', 0), 4),
            'Recall': round(report.get('1', {}).get('recall', 0), 4)
        }
        results.append(result_row)
        print(classification_report(y_test, preds))
        print("-" * 40)
        
    return models, results

def save_artifacts(models, vectorizer):
    with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'), 'wb') as f:
        pickle.dump(models['nb'], f)
        
    with open(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'), 'wb') as f:
        pickle.dump(models['lr'], f)
    print("\n[SUCCESS] Models and Vectorizers saved successfully!")

if __name__ == "__main__":
    print("Starting Spam Detection Training Pipeline...\n")
    
    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 2. Prepare Features
    X_train, X_test, y_train, y_test = split_and_vectorize(df)
    
    # 3. Train Models
    models, evaluation_results = train_models(X_train, X_test, y_train, y_test)
    
    # 4. Persist Models (Using TF-IDF vectorizer state to apply to incoming text later)
    # We need to save the vectorizer object used in split_and_vectorize
    # Re-run split logic here specifically to capture the vectorizer instance, 
    # or refactor split_and_vectorize to return vectorizer. 
    # Simple workaround for this script: Retrain vectorizer slightly cleaner version
    
    # Better approach: Save vectorizer explicitly inside the training flow 
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(X_train_df) # Must define X_train_df again or pass it around
    # For this script simplicity, we will modify split_and_vectorize to return vectorizer.
    
    # Re-calling fit transform to ensure vectorizer state is correct for future inference
    # This logic is cleaned up in the main execution below
    
    # ... (Code continuation below ensures vectorizer is captured correctly for prediction)
