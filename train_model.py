
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import clean_text

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset():
    path = os.path.join(DATA_DIR, 'spam.csv')
    if not os.path.exists(path):
        print("Downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)
        url = "https://raw.githubusercontent.com/janishar-arafat/Spam-Classification-Project/master/Datasets/dataset_spam.csv"
        import urllib.request
        urllib.request.urlretrieve(url, path)
        df = pd.read_csv(path, encoding="latin-1")
        if df.shape[1] >= 2:
            df.columns = ['Label', 'Message']
        df['Label'] = df['Label'].str.replace('no', 'ham').replace('yes', 'spam')
        df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path, encoding="latin-1")
        df.columns = ['Label', 'Message']
        df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    return df

def train_and_save():
    print("Loading data...")
    df = load_dataset()
    
    # Clean Data
    df['Clean_Mess'] = df['Message'].apply(clean_text)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Clean_Mess'], df['Label'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Models
    models = {}
    
    print("\nTraining Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    models['naive_bayes'] = nb
    
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    models['logistic_regression'] = lr
    
    # Evaluate
    print("\n--- Evaluation ---")
    for name, clf in models.items():
        pred = clf.predict(X_test_vec)
        print(f"\n{name} Performance:")
        print(classification_report(y_test, pred))
        
    # Save Artifacts
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    for name, clf in models.items():
        with open(os.path.join(MODEL_DIR, f'{name}_model.pkl'), 'wb') as f:
            pickle.dump(clf, f)
            
    print("\n✅ Models and vectorizer saved in 'models/' directory.")

if __name__ == "__main__":
    train_and_save()
