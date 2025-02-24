from flask import Flask, render_template, request, jsonify
from models.chatbot_model import MentalHealthChatbot
import logging

app = Flask(__name__)

# Configurar el registro de errores
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    handlers=[
        logging.FileHandler("error.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Crear una instancia del chatbot con el modelo fine-tuned
try:
    chatbot = MentalHealthChatbot(model_path='models/bert_emotion_model')
except Exception as e:
    logger.error(f"Error al inicializar el chatbot: {e}")
    raise

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error al renderizar index.html: {e}")
        return "Error al cargar la página de inicio.", 500

@app.route('/chatbot')
def chatbot_page():
    try:
        return render_template('chatbot.html')
    except Exception as e:
        logger.error(f"Error al renderizar chatbot.html: {e}")
        return "Error al cargar la página del chatbot.", 500

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    try:
        user_input = request.form.get('message', '').strip()
        if not user_input:
            logger.warning("Mensaje vacío recibido del usuario.")
            return jsonify({'response': "Por favor, ingresa un mensaje.", 'audio_path': None}), 400

        response_data = chatbot.generate_response(user_input)
        response_text = response_data.get('text', "Lo siento, no pude procesar tu mensaje.")
        audio_path = response_data.get('audio_path', '')

        # No es necesario verificar la existencia del archivo aquí
        return jsonify({'response': response_text, 'audio_path': audio_path})

    except Exception as e:
        logger.error(f"Error en /get_response: {e}")
        return jsonify({'response': "Lo siento, ha ocurrido un error al procesar tu solicitud.", 'audio_path': None}), 500

if __name__ == '__main__':
    app.run(debug=True)
