import unittest
import json
from flask import Flask
from main import app  # Asegúrate de que esta sea la ruta correcta al archivo principal del API

class TestAPI(unittest.TestCase):

    def setUp(self):
        # Crear una aplicación de prueba de Flask
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Crear un mensaje de prueba
        test_data = {
            "message": "¡Urgente! Verifica tu cuenta aquí: http://phishingsite.com"
        }

        # Hacer una solicitud POST al endpoint de predicción
        response = self.app.post('/predict', data=json.dumps(test_data), content_type='application/json')
        
        # Verificar que la respuesta tenga un código de estado 200
        self.assertEqual(response.status_code, 200, "El código de estado no es 200.")
        
        # Verificar que la respuesta contenga una predicción
        response_json = json.loads(response.data)
        self.assertIn("prediction", response_json, "No se encontró el campo 'prediction' en la respuesta.")

if __name__ == '__main__':
    unittest.main()
