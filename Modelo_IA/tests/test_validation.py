import unittest
from Modelo_IA.src.validate_model import validate_model

class TestModelValidation(unittest.TestCase):

    def test_validate_model(self):
        try:
            # Probar la validación del modelo
            validate_model("Modelo_IA/data/phishing_data.csv")
            # Si la función no lanza excepciones, la prueba es exitosa
            success = True
        except Exception as e:
            print(e)
            success = False
        
        self.assertTrue(success, "La validación del modelo ha fallado.")

if __name__ == '__main__':
    unittest.main()
