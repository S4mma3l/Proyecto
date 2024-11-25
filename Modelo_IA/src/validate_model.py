from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from Modelo_IA.src.preprocess import preprocess_data

def validate_model(data_path):
    """Busca los mejores parámetros para el modelo."""
    X, y, _ = preprocess_data(data_path)
    
    # Asegurarse de que las columnas sean cadenas (si aplica)
    if hasattr(X, 'columns'):
        X.columns = X.columns.astype(str)

    # Definir parámetros para Grid Search
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Define un objeto de validación cruzada con menos splits
    cv = StratifiedKFold(n_splits=3)  # 3 en lugar de 5

    # Realizar búsqueda de parámetros
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)  # Usar param_grid aquí
    grid_search.fit(X, y)
    
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor puntuación:", grid_search.best_score_)

if __name__ == "__main__":
    validate_model("Modelo_IA/data/phishing_data.csv")
