import joblib
import pandas as pd
from Modelo_IA.src.preprocess import clean_text, extract_features

def predict_message(message):
    """Predice si un mensaje es phishing o legítimo."""
    # Cargar modelo y vectorizador
    model = joblib.load("Modelo_IA/models/phishing_model.pkl")
    vectorizer = joblib.load("Modelo_IA/models/tfidf_vectorizer.pkl")
    
    # Preprocesar el mensaje
    clean_msg = clean_text(message)
    extra_features = extract_features([clean_msg])
    
    # Vectorizar texto
    tfidf_features = vectorizer.transform([clean_msg])
    
    # Combinar características
    features = pd.concat([extra_features, pd.DataFrame(tfidf_features.toarray())], axis=1)
    
    # Realizar predicción
    prediction = model.predict(features)[0]
    return "Phishing" if prediction == 1 else "Legítimo"
