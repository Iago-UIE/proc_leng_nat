from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Crear una imagen con PIL
font = ImageFont.truetype("arial.ttf", 100)
img_pil = Image.new("RGB", (300, 150), "white")
draw = ImageDraw.Draw(img_pil)
draw.text((10, 10), "Hola", font=font, fill="black")

# Convertir la imagen PIL a un array de NumPy para OpenCV
img_cv = np.array(img_pil)

# OpenCV usa BGR en lugar de RGB, por lo que se necesita convertir los colores
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
print(img_cv)
# Mostrar la imagen con OpenCV
cv2.imshow("Imagen OpenCV", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
