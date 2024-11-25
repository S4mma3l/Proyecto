import os
from Modelo_IA.src.train_model import train_and_save_model
from Modelo_IA.src.validate_model import validate_model

def execute_pipeline():
    data_path = "Modelo_IA/data/phishing_data.csv"
    model_path = "Modelo_IA/models/phishing_model.pkl"
    vectorizer_path = "Modelo_IA/models/tfidf_vectorizer.pkl"
    
    print("Validando modelo...")
    validate_model(data_path)
    
    print("Entrenando modelo...")
    train_and_save_model(data_path, model_path, vectorizer_path)
    
    print("Pipeline completado. Modelo actualizado.")

if __name__ == "__main__":
    execute_pipeline()
