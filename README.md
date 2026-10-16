# Pre-entrega 3: Sistema de Recuperación Semántica Local (RAG)

Este repositorio implementa un pipeline **RAG End-to-End** asíncrono utilizando **LangChain Expression Language (LCEL)**, **ChromaDB** como base de datos vectorial local y **Pydantic** para estructurar la respuesta final.

## 🏗️ Arquitectura del Sistema

1. **Ingesta (`ingest.py`)**:
   - Lee archivos `.txt` desde la carpeta `/data`.
   - Fragmenta el texto utilizando `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
   - Genera embeddings con `text-embedding-3-small` de OpenAI y los persiste en `./vectorstore` mediante ChromaDB.

2. **Recuperación y Generación Grounded (`main.py`)**:
   - Recupera los top 3 fragmentos más relevantes vía similitud vectorial (`Retriever`).
   - Inserta el contexto en un prompt con filtro de veracidad anti-alucinaciones.
   - Procesa la consulta de forma asíncrona (`.ainvoke()`) devolviendo un esquema `RespuestaRAG` con la respuesta y las fuentes utilizadas.

## 🚀 Instalación y Ejecución

1. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt