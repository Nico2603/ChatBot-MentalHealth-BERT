# Chatbot de Salud Mental - Versión 1.0

## Descripción del Proyecto
Este proyecto es un chatbot orientado a la salud mental que, mediante Procesamiento de Lenguaje Natural (PLN), analiza los mensajes ingresados por los usuarios (texto o audio) para predecir su estado emocional y generar respuestas de apoyo o contestaciones acordes.

Actualmente, esta versión 1.0 es una implementación básica con un conjunto limitado de emociones y respuestas. Se planea una futura versión 2.0 con mejoras en el reconocimiento y respuesta emocional, siempre considerando la complejidad de tratar temas de salud mental de manera responsable.

## Tecnologías Utilizadas
- **Python** (Flask, Transformers, PyTorch)
- **BERT (Bidirectional Encoder Representations from Transformers)**
- **Procesamiento de Lenguaje Natural (PLN)**
- **Reconocimiento de Voz y Síntesis de Texto a Voz**
- **HTML, CSS, JavaScript (Frontend)**

## Arquitectura del Chatbot
El chatbot sigue el siguiente pipeline de procesamiento:
```
-> Speech Recognition -> Natural Language Understanding -> Dialog Manager <-> Task Manager
Text-to-Speech Synthesis <- Natural Language Generation <- Dialog Manager
```
### Emociones Detectadas
El modelo ha sido entrenado para reconocer las siguientes emociones:
- FELICIDAD
- NEUTRAL
- DEPRESIÓN
- ANSIEDAD
- ESTRÉS
- EMERGENCIA
- CONFUSIÓN
- IRA
- MIEDO
- SORPRESA
- DISGUSTO

Se ha utilizado un dataset de 500 muestras para cada una de estas emociones.

## Estructura del Proyecto
```
ChatBot/
├── conversations/
├── data/
│   ├── emotion_dataset.csv
├── models/
│   ├── bert_emotion_model/
│       ├── checkpoint-1600
│       ├── checkpoint-1650
│       ├── config.json
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       ├── vocab.txt
│   ├── chatbot_model.py
│   ├── responses.json
├── static/
│   ├── audio/
│   ├── css/
│   │   ├── styles.css
│   ├── img/
│   ├── js/
│       ├── scripts.js
├── templates/
│   ├── chatbot.html
│   ├── index.html
├── app.py
├── chatbot.log
├── error.log
├── requirements.txt
├── train_model.py
```

## Instalación y Configuración
### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/ChatBot-Clean.git
cd ChatBot-Clean
```

### 2. Crear un entorno virtual y activarlo
```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
python app.py
```

La aplicación se ejecutará en `http://127.0.0.1:5000/`.

## Flujo de Uso de los Archivos en el Proyecto
1. **Cargar el Modelo**: Se utilizan los pesos del modelo almacenados en `model.safetensors` junto con `config.json` para definir la estructura.
2. **Tokenización**: Se convierten las entradas de texto en tokens que BERT puede procesar usando `tokenizer.json` y `vocab.txt`.
3. **Inferencia**: Se analiza el mensaje del usuario, se predice la emoción y se genera una respuesta en base al dataset `emotion_dataset.csv`.

## Notas Finales
- Esta versión es experimental y no sustituye asesoramiento profesional en salud mental.
- Se recomienda seguir desarrollando y refinando el modelo para mejorar su precisión y amplitud de respuestas.

**Autor:** Nicolás Ceballos Brito  
**Contacto:** nicolasceballosbrito@gmail.com