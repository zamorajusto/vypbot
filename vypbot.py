import os
import shutil
import time
import pandas as pd
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- RUTAS DINÁMICAS (A prueba de balas) ---
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_PDF = os.path.join(CARPETA_ACTUAL, "condiciones_met99.pdf")
RUTA_EXCEL = os.path.join(CARPETA_ACTUAL, "preguntas_frecuentes.xlsx")
CARPETA_DB = os.path.join(CARPETA_ACTUAL, "db_met99")

def limpiar_base_datos():
    if os.path.exists(CARPETA_DB):
        print(f"🧹 Limpiando memoria anterior en: {CARPETA_DB}")
        try:
            shutil.rmtree(CARPETA_DB)
            time.sleep(1)
        except Exception as e:
            print(f"   ⚠️ Windows bloqueó el archivo: {e}. Cierra VS Code o terminales y borra 'db_met99' manual.")
            return False
    return True

def crear_base_datos():
    print("\n🚀 INICIANDO CONSTRUCCIÓN DEL CEREBRO DE VypBot (MODO QUIRÚRGICO)...")
    if not limpiar_base_datos(): return 

    print("\n--- 1. LEYENDO DOCUMENTOS ---")
    docs = []

    # --- LECTURA AVANZADA DE PDF CON TABLAS (PDFPLUMBER) ---
    if os.path.exists(RUTA_PDF):
        print(f"   📄 Analizando PDF y reconstruyendo tablas...")
        try:
            with pdfplumber.open(RUTA_PDF) as pdf:
                for num_pagina, pagina in enumerate(pdf.pages):
                    # 1. Extraer texto normal
                    texto_pagina = pagina.extract_text() or ""
                    
                    # 2. Extraer y formatear TABLAS
                    tablas = pagina.extract_tables()
                    if tablas:
                        texto_pagina += "\n\n--- TABLAS DE REFERENCIA ---\n"
                        for tabla in tablas:
                            for fila in tabla:
                                # Limpia celdas vacías y quita saltos de línea raros dentro de la tabla
                                fila_limpia = [str(celda).replace('\n', ' ').strip() if celda else "N/A" for celda in fila]
                                # Une las columnas con un separador claro para la IA
                                texto_pagina += " | ".join(fila_limpia) + "\n"
                            texto_pagina += "----------------------------\n"
                    
                    # 3. Guardar la página si tiene contenido
                    if texto_pagina.strip():
                        docs.append(Document(
                            page_content=texto_pagina, 
                            metadata={"source": "PDF Met99", "page": num_pagina + 1}
                        ))
            print(f"      -> {len(pdf.pages)} páginas analizadas con éxito.")
        except Exception as e:
            print(f"      ❌ Error leyendo PDF con pdfplumber: {e}")
    else:
        print(f"   ❌ NO SE ENCUENTRA EL PDF: {RUTA_PDF}")

    # --- LECTURA DE EXCEL ---
    if os.path.exists(RUTA_EXCEL):
        print(f"   📊 Procesando Excel de FAQs...")
        try:
            df = pd.read_excel(RUTA_EXCEL)
            for index, row in df.iterrows():
                if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    texto = f"PREGUNTA: {row.iloc[0]}\nRESPUESTA: {row.iloc[1]}"
                    docs.append(Document(page_content=texto, metadata={"source": "Excel FAQ"}))
            print(f"      -> {len(df)} preguntas estructuradas.")
        except Exception as e:
            print(f"      ❌ Error en Excel: {e}")

    if not docs:
        print("\n❌ Cancelado: No hay documentos.")
        return

    # --- FRAGMENTACIÓN Y GUARDADO ---
    print("\n--- 2. FRAGMENTANDO TEXTO ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.split_documents(docs)
    
    print("\n--- 3. GUARDANDO EN CHROMA DB ---")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=CARPETA_DB)
        print(f"\n✅ ¡ÉXITO! Base de datos lista con tablas legibles.")
    except Exception as e:
        print(f"\n❌ Error guardando: {e}")
    
if __name__ == "__main__":
    crear_base_datos()