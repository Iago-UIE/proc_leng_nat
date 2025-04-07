import streamlit as st
import ollama
from PIL import Image
import io
import base64
from streamlit.components.v1 import html
import pandas as pd
from datetime import datetime

# Función para crear un componente HTML que permita pegar solo imágenes
def image_upload_clipboard():
    html_code = """
    <div id="clipboard-area" 
         style="width: 100%; height: 300px; border: 2px dashed #aaa; border-radius: 8px; 
                text-align: center; display: flex; align-items: center; justify-content: center;
                transition: all 0.3s ease; cursor: pointer; background: #f9f9f9;"
         tabindex="0">
        <p id="status-text">📎 Pega la imagen aquí (Ctrl+V)</p>
    </div>
    
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        const clipboardArea = document.getElementById('clipboard-area');
        const statusText = document.getElementById('status-text');
        
        // Configurar foco al hacer clic
        clipboardArea.addEventListener('click', () => clipboardArea.focus());
        
        // Efectos visuales de foco
        clipboardArea.addEventListener('focus', () => {
            clipboardArea.style.borderColor = '#4CAF50';
            clipboardArea.style.background = '#f5fff5';
        });
        
        clipboardArea.addEventListener('blur', () => {
            clipboardArea.style.borderColor = '#aaa';
            clipboardArea.style.background = '#f9f9f9';
        });

        // Manejar pegado
        clipboardArea.addEventListener('paste', async (event) => {
            const items = event.clipboardData.items;
            const fileItem = [...items].find(item => item.kind === 'file' && item.type.startsWith('image/'));
            
            if (!fileItem) {
                statusText.innerHTML = "❌ No se encontró imagen en el portapapeles";
                clipboardArea.style.borderColor = '#ff4444';
                setTimeout(() => {
                    statusText.innerHTML = "📎 Pega la imagen aquí (Ctrl+V)";
                    clipboardArea.style.borderColor = '#aaa';
                }, 2000);
                return;
            }
            
            try {
                const blob = fileItem.getAsFile();
                const reader = new FileReader();
                
                reader.onload = () => {
                    const base64Image = reader.result.split(',')[1];
                    statusText.innerHTML = "✅ ¡Imagen pegada correctamente!";
                    clipboardArea.style.borderColor = '#4CAF50';
                    sendToStreamlit(base64Image);
                };
                
                reader.readAsDataURL(blob);
            } catch (error) {
                statusText.innerHTML = "❌ Error al procesar la imagen";
                console.error('Error:', error);
            }
        });
    });
    
    function sendToStreamlit(base64Image) {
        const message = {
            type: "streamlit:setValue",
            key: "image_base64",
            value: base64Image
        };
        window.parent.postMessage(message, "*");
    }
    </script>
    """
    html(html_code, height=320)

# Configuración de la aplicación
st.set_page_config(page_title="PolishMyPitch", layout="centered")
st.title("📝 PolishMyPitch - Perfeccionador de Texto con Estilo")
st.markdown("Utiliza **Ctrl+V** para pegar una imagen con texto o ingresa el texto manualmente.")

# Crear dos columnas para imagen y texto
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen (Ctrl+V)")
    image_upload_clipboard()
    # Intentamos leer la imagen del estado de sesión
    image_base64 = st.session_state.get("image_base64")
    if image_base64:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Imagen cargada", use_column_width=True)
        # Usar llava para extraer el texto de la imagen
        response_img = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_data},
                        {"type": "text", "text": "Extrae todo el texto visible de esta imagen."}
                    ]
                }
            ]
        )
        extracted_text = response_img['message']['content']
        st.subheader("Texto extraído:")
        st.write(extracted_text)
    else:
        extracted_text = ""

with col2:
    st.subheader("Texto manual")
    user_input = st.text_area("Pega el texto manualmente:", height=300)

# Selector de estilo y ajustes avanzados
estilo = st.selectbox("🧠 ¿Qué idioma prefieres?", ["Español", "Inglés", "Chino", "Indio"])
with st.expander("⚙️ Configuración avanzada"):
    temperatura = st.slider("Creatividad (temperatura)", 0.0, 1.5, 0.7, step=0.1)
    tokens_maximos = st.slider("Longitud máxima de respuesta (tokens)", 50, 100000, 400, step=50)

# Acción al presionar el botón
if st.button("✨ Generar texto"):
    if not extracted_text and not user_input.strip():
        st.warning("Por favor, pega una imagen o escribe un texto.")
    else:
        with st.spinner("Procesando con el modelo de lenguaje..."):
            # Se prioriza el texto extraído de la imagen, si existe; de lo contrario, se usa el texto manual
            texto_prompt = extracted_text if extracted_text else user_input
            prompt = "Por favor, no respondas con texto normal, solo debes responder con un sólo número de teléfono que aparece aqui: \n'''{texto_prompt}'''"
            rol = "Sé una máquina super especializada en sacar números de teléfono de texto"

            if estilo.lower() == "español":
                prompt += "La respuesta debe estar en español:"
                rol += "Hablas español."
            elif estilo.lower() == "factura":
                prompt += "La respuesta debe estar en chino:"
                rol += "Eres chino"
            elif estilo.lower() == "Inglés":
                prompt += "La respuesta debe estar en inglés:"
                rol = "Eres inglés."
                
            elif estilo.lower() == "indio":
                prompt += "La respuesta debe estar en indio:"
                rol += "Eres indio."
            else:
                rol = "Sé un asistente útil."
            prompt = f"{rol}\n{prompt}"
            response = ollama.chat(
                model='llama3',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperatura,
                    "num_predict": tokens_maximos
                }
            )
            resultado = response['message']['content']
            st.subheader("📌 Texto generado:")
            st.write(resultado)
            
            # Crear DataFrame para exportar a CSV
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            datos = {
                "Fecha": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Estilo": [estilo],
                "Texto_Original": [texto_prompt],
                "Texto_Generado": [resultado],
                "Temperatura": [temperatura],
                "Tokens_Maximos": [tokens_maximos]
            }
            df = pd.DataFrame(datos)
            
            # Convertir DataFrame a CSV
            csv = df.to_csv(index=False, encoding='utf-8')
            
            # Mostrar botones de descarga
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Descargar como .txt",
                    data=resultado,
                    file_name=f"texto_generado_{marca_tiempo}.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="📊 Descargar como .csv",
                    data=csv,
                    file_name=f"resultados_{marca_tiempo}.csv",
                    mime="text/csv"
                )