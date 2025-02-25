import os
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

# Establecer la semilla para garantizar reproducibilidad
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Función para cargar datos (simplificada para UTF-8)
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    return data

# Función para normalizar texto, manteniendo caracteres especiales
def normalize_text(text):
    if isinstance(text, str):
        return text.strip().upper()
    return text

# Función para limpiar y preparar los datos
def clean_and_prepare_data(data):
    data = data.copy()
    # Eliminar filas con valores nulos
    data = data.dropna(subset=['text', 'label'])
    # Normalizar las etiquetas
    data['label'] = data['label'].apply(normalize_text)
    # Definir las etiquetas esperadas
    emotion_labels = ['FELICIDAD', 'NEUTRAL', 'DEPRESIÓN', 'ANSIEDAD', 'ESTRÉS',
                      'EMERGENCIA', 'CONFUSIÓN', 'IRA', 'MIEDO', 'SORPRESA', 'DISGUSTO']
    # Filtrar solo las etiquetas conocidas
    data = data[data['label'].isin(emotion_labels)]
    # Crear el mapeo de etiquetas
    label_to_id = {label: idx for idx, label in enumerate(emotion_labels)}
    data['label'] = data['label'].map(label_to_id)
    # Verificar que no haya valores NaN
    if data['label'].isna().any():
        data = data.dropna(subset=['label'])
    data['label'] = data['label'].astype(int)
    return data, emotion_labels, label_to_id

# Función para dividir los datos
def split_data(data):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'], data['label'],
        test_size=0.2,
        stratify=data['label'],
        random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels

# Función para calcular los pesos de clase
def get_class_weights(labels):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)

# Función para tokenizar los datos (sin padding, ya que lo maneja el data collator)
def tokenize_data(tokenizer, texts, labels):
    dataset = Dataset.from_dict({'text': texts.tolist(), 'label': labels.tolist()})
    dataset = dataset.map(lambda batch: tokenizer(batch['text'], truncation=True, max_length=128), batched=True)
    return dataset

# Función de pérdida personalizada que incorpora los pesos de clase
def custom_loss(labels, logits):
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    return loss_fct(logits, labels)

# Clase CustomTrainer para usar la función de pérdida personalizada
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").to(model.device)
        # Realizar el forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Calcular la pérdida personalizada
        loss = custom_loss(labels, logits)
        return (loss, outputs) if return_outputs else loss

# Función para calcular métricas de evaluación
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Función para predecir la etiqueta de un texto dado
def predict(text):
    # Tokenizar el texto
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Realizar la predicción
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        label = id_to_label.get(predicted_class, "Etiqueta desconocida")
    return label

if __name__ == '__main__':
    # Configurar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}")

    # Ruta del archivo CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'data', 'emotion_dataset.csv')

    # Paso 1: Cargar y preparar los datos
    data = load_data(input_file)
    data, emotion_labels, label_to_id = clean_and_prepare_data(data)
    id_to_label = {v: k for k, v in label_to_id.items()}

    # Paso 2: Dividir los datos
    train_texts, val_texts, train_labels, val_labels = split_data(data)

    # Paso 3: Calcular los pesos de clase
    class_weights = get_class_weights(train_labels).to(device)

    # Paso 4: Configurar el tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

    # Paso 5: Tokenizar los datos
    train_dataset = tokenize_data(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_data(tokenizer, val_texts, val_labels)

    # Paso 6: Configurar el data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Paso 7: Configurar el modelo
    model = BertForSequenceClassification.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-cased',
        num_labels=len(emotion_labels)
    )

    # Paso 8: Configurar el entrenamiento
    training_args = TrainingArguments(
        output_dir='./models/bert_emotion_model',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        lr_scheduler_type='linear',
        warmup_steps=500,
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to="none"
    )

    # Paso 9: Crear el entrenador personalizado
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Paso 10: Entrenar el modelo
    trainer.train()

    # Paso 11: Guardar el modelo y el tokenizer
    trainer.save_model('./models/bert_emotion_model')
    tokenizer.save_pretrained('./models/bert_emotion_model')

    # Paso 12: Guardar los mapeos de etiquetas
    with open('./models/bert_emotion_model/label_to_id.json', 'w') as f:
        json.dump(label_to_id, f)
    with open('./models/bert_emotion_model/id_to_label.json', 'w') as f:
        json.dump(id_to_label, f)

    print("\nModelo entrenado y guardado exitosamente.")

    # Paso 13: Probar el modelo con un ejemplo
    sample_text = "Me siento muy feliz hoy"
    print(f"Texto: {sample_text}")
    print(f"Predicción: {predict(sample_text)}")