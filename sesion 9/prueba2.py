import torch
from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor

# Carga del modelo y procesador
model = ColQwen2.from_pretrained(
    "Metric-AI/colqwen2.5-3b-multilingual",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # O "mps" si usas Apple Silicon
).eval()
processor = ColQwen2Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")

# Entradas: imagen y consulta de texto
image = Image.open("factura.png").convert("RGB")
queries = [
    "Is attention really all you need?"
]

# Procesa la imagen y la consulta
batch_images = processor.process_images(image).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Realiza la inferencia usando los métodos específicos para cada modalidad
with torch.no_grad():
    image_embeddings = model.get_image_embeddings(**batch_images)
    query_embeddings = model.get_text_embeddings(**batch_queries)

# Calcula la similitud o puntuación entre las representaciones
scores = processor.score_multi_vector(query_embeddings, image_embeddings)

print(scores)
