from pdf2image import convert_from_path

# Convertir PDF a imágenes
images = convert_from_path("factura.pdf", dpi=300)  # DPI alto para mejor calidad

# Guardar cada página como PNG
for i, img in enumerate(images):
    img.save(f"pagina_{i+1}.png", "PNG")

print("Conversión completada.")

