import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#TOKEN Hugging Face
os.environ["HF_TOKEN"] = st.secrets["HF_API_KEY"]

# --- 1. CONFIGURACIÓN DEL SISTEMA ---
# Pon tu clave real de Groq aquí
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MODELO = "llama-3.3-70b-versatile"

# ¡IMPORTANTE! Usa la ruta absoluta exacta donde se guardó tu base de datos
#CARPETA_DB = r"c:\Users\JQ Hector Zamora\OneDrive\Documentos\ProyectosPy\vypbot\db_met99"
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CARPETA_DB = os.path.join(CARPETA_ACTUAL, "db_met99")

# --- 2. CONFIGURACIÓN DE LA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="Asistente Met99 | VPC",
    page_icon="🛡️",
    layout="centered"
)

# --- 3. CARGA DE MEMORIA Y CEREBRO (Con Caché para que sea rápido) ---
# Usamos cache_resource para que no recargue la base de datos en cada pregunta
@st.cache_resource
def cargar_motor():
    # Cargar Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Conectar a la base de datos
    vector_db = Chroma(persist_directory=CARPETA_DB, embedding_function=embedding_model)
    # Conectar a Groq
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODELO, temperature=0.3)
    
    return vector_db, llm

# Inicializamos los motores
try:
    vector_db, llm = cargar_motor()
except Exception as e:
    st.error(f"Error crítico al cargar el sistema: {e}")
    st.stop()

# --- 4. MEMORIA DEL CHAT (Historial en pantalla) ---
# Si es la primera vez que abre, creamos un mensaje de bienvenida
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": "¡Hola, equipo! 👋 Soy VyPBot 🤖, tu nuevo compañero y asistente virtual en Vida y Pensiones. Estoy aquí para ayudarte a dominar todo sobre Met99. ¿Qué duda resolvemos hoy?"}
    ]

# --- 5. INTERFAZ GRÁFICA ---
st.title("🛡️ Asistente Inteligente Met99")
st.caption("Resuelve tus dudas sobre condiciones, coberturas y reglas de negocio al instante.")
st.divider()

# Dibujar todos los mensajes guardados en el historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. LÓGICA DE PREGUNTA Y RESPUESTA ---
pregunta_usuario = st.chat_input("Escribe tu pregunta sobre Met99 aquí...")

if pregunta_usuario:
    # 1. Mostrar la pregunta del usuario
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    # 2. Buscar y Pensar
    with st.chat_message("assistant"):
        with st.spinner("Buscando en manuales y analizando..."):
            try:
                # Buscar en la base de datos
                docs = vector_db.similarity_search(pregunta_usuario, k=4)
                
                if not docs:
                    respuesta_final = "No encontré información en los documentos."
                    st.markdown(respuesta_final)
                else:
                    # Unir el contexto
                    contexto_texto = "\n\n".join([f"- {d.page_content}" for d in docs])
                    
                    # --- NUEVO: BOTÓN DE RAYOS X ---
                    # Esto te mostrará en pantalla exactamente qué leyó la IA
                    #with st.expander("🔍 Ver fragmentos encontrados en los manuales"):
                        #st.info(contexto_texto)
                    
                    # Armar el mensaje
                    mensaje_sistema = f"""
                    Eres VyPBot, el asistente virtual estrella y compañero de equipo en la promotoría Vida y Pensiones.
                    Tu personalidad es empática, enérgica, profesional y siempre dispuesta a ayudar (puedes usar emojis ocasionalmente para ser amigable).
                    
                    Tu misión principal es apoyar a los agentes de seguros a resolver dudas sobre el producto "Met99" de MetLife, el cual es el único producto en el que nos enfocamos.
                    
                    REGLAS:
                    1. Responde a la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
                    2. Si la respuesta no está en el contexto, di amablemente: "¡Ups! No tengo esa información en mis manuales actuales, pero puedes consultarlo con alguien del equipo."
                    3. Mantén un tono motivador que impulse las ventas y la confianza del agente.
                                        
                    CONTEXTO ENCONTRADO:
                    {contexto_texto}
                    """
                    
                    # Llamar a Groq
                    respuesta_ia = llm.invoke([
                        ("system", mensaje_sistema),
                        ("human", pregunta_usuario)
                    ])
                    respuesta_final = respuesta_ia.content
                    
                    # Mostrar la respuesta final
                    st.markdown(respuesta_final)
                
                # Guardar en historial
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_final})

            except Exception as e:
                st.error(f"Hubo un error al procesar la respuesta: {e}")

st.markdown("---")
st.markdown("🎯 Desarrollado por Enrique Zamora®️ v1.1 IA IA IA 🚀")