import unittest
import pandas as pd
from Modelo_IA.src.preprocess import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        # Suponiendo que preprocess_data devuelve tres valores: X, y, y el data frame original
        X, y, _ = preprocess_data("Modelo_IA/data/phishing_data.csv")
        
        # Verificar que X y y no sean vacíos
        self.assertGreater(len(X), 0, "El conjunto de características X está vacío.")
        self.assertGreater(len(y), 0, "El conjunto de etiquetas y está vacío.")
        
        # Verificar que las columnas de X sean de tipo string
        self.assertTrue(all(isinstance(col, str) for col in X.columns), "Algunas columnas no son cadenas.")

    def test_clean_text_function(self):
        """Test para verificar que la función de limpieza funciona correctamente."""
        # Crear una entrada de texto de prueba
        raw_text = "¡Compra ahora! https://www.fakeurl.com"
        clean_text = preprocess_data.clean_text(raw_text)
        
        # Verificar que la URL haya sido eliminada
        self.assertNotIn("http", clean_text, "La URL no ha sido eliminada correctamente.")

if __name__ == '__main__':
    unittest.main()
