import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

# Cargar el modelo
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

# Cargar la imagen de la factura
image = Image.open("factura.png").convert("RGB")

# Consulta sobre el NIF del proveedor
query = "¿Cuál es el NIF del proveedor en esta factura?"

# Procesar la imagen y la consulta
batch_image = processor.process_images([image]).to(model.device)
batch_query = processor.process_queries([query]).to(model.device)

# Paso hacia adelante
with torch.no_grad():
    image_embedding = model(**batch_image)
    query_embedding = model(**batch_query)

# Calcular la similitud
score = processor.score_multi_vector(query_embedding, image_embedding)

print("Score de similitud:", score)
