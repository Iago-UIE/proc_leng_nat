import onnxruntime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Cargar el modelo ONNX
session = onnxruntime.InferenceSession("modelo.onnx")

# Función para cargar y preprocesar una imagen de prueba
def load_and_preprocess_image(file_path):
    img = load_img(file_path, color_mode='grayscale', target_size=(200, 50))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión de lote
    return img_array.astype(np.float32)

# Cargar y preprocesar una imagen de ejemplo
image_path = "files/file_2.png"  # Reemplaza con una ruta de imagen válida
input_data = load_and_preprocess_image(image_path)

# Realizar la inferencia ONNX
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
onnx_prediccion = session.run([output_name], {input_name: input_data})[0]

# Decodificar la predicción (usando la misma lógica que en el entrenamiento)
def decode_prediction(prediccion, char_to_num):
    num_to_char = {v: k for k, v in char_to_num.items()}
    input_length = np.ones(prediccion.shape[0]) * prediccion.shape[1]
    results = tf.keras.backend.ctc_decode(prediccion, input_length, greedy=True)[0][0].numpy()
    return "".join([num_to_char.get(c, '') for c in results[0] if c != 0])

# Cargar el mapeo de caracteres (necesario para la decodificación)
import pandas as pd
df = pd.read_csv('metadata.csv')
y_texts = df["Text"].astype(str).fillna("")
characters = sorted(set("".join(y_texts)))
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}

# Decodificar la predicción ONNX
predicted_text = decode_prediction(onnx_prediccion, char_to_num)
print("Texto predicho por ONNX:", predicted_text)