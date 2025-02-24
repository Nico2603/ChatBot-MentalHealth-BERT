import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import re
from datetime import datetime
import os
import logging
from typing import Tuple, Dict, Any
import json
import pyttsx3

class MentalHealthChatbot:
    def __init__(self, model_path: str = 'models/bert_emotion_model'):
        """
        Inicializa el chatbot con el modelo BERT fine-tuned y configuraciones necesarias.
        Args:
            model_path: Ruta al modelo fine-tuned
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Configuración del logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('chatbot.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        try:
            self.logger.info("Cargando el tokenizador y el modelo BERT fine-tuned...")

            # Crear carpeta para guardar historiales si no existe
            os.makedirs('conversations', exist_ok=True)

            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)

            # Cargar respuestas predefinidas
            self.load_responses()

            # Inicializar el historial de conversación
            self.conversation_history = []

            self.logger.info("Chatbot inicializado correctamente.")
        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {str(e)}")
            raise e

    def load_responses(self):
        """Carga las respuestas predefinidas desde un archivo JSON."""
        try:
            with open('models/responses.json', 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            self.logger.info("Respuestas cargadas desde 'responses.json'.")
        except FileNotFoundError:
            self.logger.error("Archivo 'responses.json' no encontrado. Asegúrate de que el archivo existe en la ruta especificada.")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error al decodificar 'responses.json': {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto de entrada."""
        try:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error al preprocesar el texto: {str(e)}")
            return text

    def detect_emergency(self, text: str) -> bool:
        """Detecta si el mensaje indica una emergencia de salud mental."""
        try:
            emergency_keywords = [
                'suicidar', 'morir', 'muerte', 'matar', 'dolor',
                'ayuda', 'emergencia', 'crisis', 'grave'
            ]
            return any(keyword in text.lower() for keyword in emergency_keywords)
        except Exception as e:
            self.logger.error(f"Error al detectar emergencia: {str(e)}")
            return False

    def get_emotion_prediction(self, text: str) -> Tuple[str, float]:
        """Predice la emoción del texto usando el modelo fine-tuned."""
        # Asegúrate de que el orden de las etiquetas coincide con el del entrenamiento
        emotion_labels = ['FELICIDAD', 'NEUTRAL', 'DEPRESIÓN', 'ANSIEDAD', 'ESTRÉS',
                          'EMERGENCIA', 'CONFUSIÓN', 'IRA', 'MIEDO', 'SORPRESA', 'DISGUSTO']

        try:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()

            emotion = emotion_labels[predicted_class]
            self.logger.info(f"Emoción predicha: {emotion} con confianza {confidence:.2f}")
            return emotion, confidence

        except Exception as e:
            self.logger.error(f"Error en la predicción de emoción: {str(e)}")
            return 'CONFUSIÓN', 0.0

    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Genera una respuesta basada en el input del usuario."""
        try:
            # Preprocesar texto
            processed_text = self.preprocess_text(user_input)
            self.logger.info(f"Texto procesado: {processed_text}")

            # Verificar emergencia
            if self.detect_emergency(processed_text):
                emotion = 'EMERGENCIA'
                confidence = 1.0
                self.logger.info("Emergencia detectada en el mensaje del usuario.")
            else:
                # Predecir emoción
                emotion, confidence = self.get_emotion_prediction(processed_text)

            # Seleccionar respuesta
            responses = self.responses.get(emotion, self.responses.get('CONFUSIÓN', ["Lo siento, no he entendido tu mensaje."]))

            response = np.random.choice(responses)
            self.logger.info(f"Respuesta seleccionada: {response}")

            # Generar audio
            audio_path = self.generate_audio(response)

            # Actualizar historial
            self.update_conversation_history(user_input, response, emotion)

            # Guardar historial después de actualizar
            self.save_conversation_history()

            return {
                'text': response,
                'audio_path': audio_path,
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error al generar respuesta: {str(e)}")
            return {
                'text': "Lo siento, ha ocurrido un error. ¿Podrías intentarlo de nuevo?",
                'audio_path': None,
                'emotion': 'ERROR',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def generate_audio(self, text: str) -> str:
        """Genera el audio para la respuesta y devuelve la URL accesible para el cliente."""
        try:
            filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
            file_path = os.path.join('static', 'audio', filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            engine = pyttsx3.init()

            # Configurar la voz en español (ajusta el índice o usa el id de la voz)
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'Spanish' in voice.name or 'Español' in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
            else:
                self.logger.warning("No se encontró una voz en español. Usando la voz predeterminada.")

            # Configurar la velocidad del habla si es necesario
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 50)  # Ajusta el valor según tus necesidades

            # Guardar el audio en el archivo especificado
            engine.save_to_file(text, file_path)
            engine.runAndWait()

            self.logger.info(f"Audio generado y guardado en {file_path}")

            # Devolver la ruta relativa que el cliente puede usar
            return f"/static/audio/{filename}"
        except Exception as e:
            self.logger.error(f"Error al generar audio: {str(e)}")
            return None

    def update_conversation_history(self, user_input: str, response: str, emotion: str):
        """Actualiza el historial de conversación."""
        try:
            self.conversation_history.append({
                'user_input': user_input,
                'response': response,
                'emotion': emotion,
                'timestamp': datetime.now().isoformat()
            })

            # Mantener solo las últimas 10 conversaciones
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)

            self.logger.info("Historial de conversación actualizado.")
        except Exception as e:
            self.logger.error(f"Error al actualizar el historial de conversación: {str(e)}")

    def save_conversation_history(self):
        """Guarda el historial de conversación en un archivo."""
        try:
            filename = f"conversations/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Historial de conversación guardado en {filename}")
        except Exception as e:
            self.logger.error(f"Error al guardar el historial: {str(e)}")
