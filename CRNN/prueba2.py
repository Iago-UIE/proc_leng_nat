import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Tamaño de imagen: 200x50, 1 canal (escala de grises)
image_shape = (200, 50, 1)

# 1. Cargar el DataFrame
df = pd.read_csv('metadata.csv')
df["FileName"] = "files/" + df["FileName"]

# Extraer etiquetas de texto
y_texts = df["Text"].astype(str).fillna("")

# 2. Codificar caracteres a números
# Extraer caracteres únicos
characters = sorted(set("".join(y_texts)))
# Mapeo: asignar cada carácter a un número empezando en 1
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
# Reservar: 0 para padding y (num_classes - 1) para blank
num_classes = len(characters) + 2  # +2 por padding (0) y blank
blank_token = num_classes - 1

# Función para convertir texto a secuencia numérica con padding
def text_to_sequence(text, max_len):
    sequence = [char_to_num.get(char, 0) for char in text]
    sequence += [0] * (max_len - len(sequence))
    return sequence[:max_len]

# 3. Longitud máxima de etiqueta
max_label_length = max(len(text) for text in y_texts)

# 4. Convertir etiquetas a secuencias numéricas
y_sequences = np.array([text_to_sequence(text, max_label_length) for text in y_texts])

# 5. Dividir en entrenamiento y prueba
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    df["FileName"].tolist(), y_sequences, test_size=0.2, random_state=42)

# Convertir etiquetas a int32 (necesario para CTC)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# 6. Función para cargar y preprocesar imágenes
def load_image(file_path):
    img = load_img(file_path, color_mode='grayscale', target_size=(200, 50))
    img_array = img_to_array(img) / 255.0  # Normalizar entre 0 y 1
    return img_array

# Cargar imágenes en memoria
X_train = np.array([load_image(fp) for fp in X_train_paths])
X_test = np.array([load_image(fp) for fp in X_test_paths])

# 7. Construcción del modelo OCR (CNN + RNN + CTC Loss)
inputs = tf.keras.layers.Input(shape=image_shape)

# Capas convolucionales para extraer características
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # Reduce a (100, 25, 32)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 1))(x)  # Reduce a (50, 25, 64)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

# Reestructurar para RNN
seq_length = x.shape[1] * x.shape[2]  # 50 * 25 = 1250
x = tf.keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)  # (50, 25*128)

# Verificar que la longitud de la secuencia sea suficiente
print("Longitud de secuencia:", x.shape[1])
assert x.shape[1] > max_label_length, f"La longitud de secuencia ({x.shape[1]}) debe ser mayor que max_label_length ({max_label_length})"

# Capas LSTM bidireccionales
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)

# Capa de salida con softmax
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Crear el modelo
model = tf.keras.models.Model(inputs, outputs)

# 8. Función de pérdida CTC
def ctc_loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    input_length = tf.fill([batch_size, 1], tf.shape(y_pred)[1])  # Longitud de la secuencia predicha
    label_length = tf.cast(tf.math.count_nonzero(y_true, axis=1, keepdims=True), tf.int32)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# 9. Compilar modelo
model.compile(optimizer='adam', loss=ctc_loss)

# 10. Entrenar el modelo con EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# 11. Evaluar el modelo
test_loss = model.evaluate(X_test, y_test)
print("Pérdida en test:", test_loss)

# 12. Predicción de una palabra
ejemplo = np.expand_dims(X_train[0], axis=0)
prediccion = model.predict(ejemplo)

# Decodificar usando ctc_decode
input_length = np.ones(prediccion.shape[0]) * prediccion.shape[1]
results = tf.keras.backend.ctc_decode(prediccion, input_length, greedy=True)[0][0].numpy()

# Convertir la secuencia numérica a texto, excluyendo padding (0) y blank token
num_to_char = {v: k for k, v in char_to_num.items()}
predicted_text = "".join([num_to_char.get(c, '') for c in results[0] if c > 0 and c != blank_token])
print("Palabra predicha:", predicted_text)

# 13. Conversión a ONNX
import tf2onnx
spec = (tf.TensorSpec((None, 200, 50, 1), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
output_path = "modelo.onnx"
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
print("Modelo ONNX guardado en:", output_path)