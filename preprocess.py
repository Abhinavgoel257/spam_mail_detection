import re
import string
import nltk
from nltk.corpus import stopwords

# Download required NLTK data (runs only once usually)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.data.path.append(os.path.dirname(__file__))

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text by converting to lowercase, removing non-alphabetic 
    characters, and removing stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http://... or https://...)
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    text = url_pattern.sub('', text)
    
    # Remove punctuation and digits
    # Keep only alphabets and spaces
    text = re.sub(f"[^{re.escape(string.ascii_letters)}\s]", '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [w for w in tokens if not w in STOP_WORDS]
    
    # Join back to string
    return " ".join(tokens)
