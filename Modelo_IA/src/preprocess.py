import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """Limpia y normaliza el texto."""
    # Normalizar URLs obfuscadas
    text = re.sub(r"hxxp://", "http://", text)
    # Quitar caracteres especiales
    text = re.sub(r"[^\w\s]", "", text)
    # Convertir a minúsculas
    text = text.lower()
    return text

def extract_features(texts):
    """Extrae características adicionales del texto."""
    features = []
    for text in texts:
        features.append({
            "contains_link": int("http" in text or "www" in text),
            "length": len(text),
            "num_words": len(text.split())
        })
    return pd.DataFrame(features)

def preprocess_data(file_path):
    """Carga, limpia y vectoriza los datos."""
    data = pd.read_csv(file_path)
    print(data.head())  # Verifica el contenido del archivo
    data["clean_text"] = data["message"].apply(clean_text)
    
    # Extraer características adicionales
    extra_features = extract_features(data["clean_text"])
    
    # Vectorizar texto con TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(data["clean_text"])
    
    # Combinar características
    X = pd.concat([extra_features, pd.DataFrame(tfidf_matrix.toarray())], axis=1)
    y = data["label"]
    
    return X, y, vectorizer