from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from Modelo_IA.src.preprocess import preprocess_data

def train_and_save_model(data_path, model_path, vectorizer_path):
    """Entrena el modelo y guarda el modelo y el vectorizador."""
    X, y, vectorizer = preprocess_data(data_path)
    
    # Asegurarse de que las columnas sean cadenas (si aplica)
    if hasattr(X, 'columns'):
        X.columns = X.columns.astype(str)
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=None)
    model.fit(X, y)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo y vectorizador
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == "__main__":
    train_and_save_model("Modelo_IA/data/phishing_data.csv", "Modelo_IA/models/phishing_model.pkl", "Modelo_IA/models/tfidf_vectorizer.pkl")
