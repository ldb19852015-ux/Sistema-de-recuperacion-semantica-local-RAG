import os

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIRECTORY = "./vectorstore"
DATA_PATH = "./data"


def indexar_documentos():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")

    print("📁 Cargando documentos desde la carpeta /data...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    if not docs:
        print("❌ No se encontraron documentos en la carpeta /data.")
        return

    print(f"📄 Documentos cargados: {len(docs)}")

    # Chunking estratégico según la consigna (mínimo 500 con overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Se generaron {len(chunks)} fragmentos (chunks).")

    # Embeddings pasando cliente HTTP con verify=False para evitar el bloqueo de red/SSL
    http_client = httpx.Client(verify=False)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        http_client=http_client,
    )

    print("💾 Guardando embeddings en ChromaDB local (./vectorstore)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    print("✅ Ingesta completada con éxito.")


if __name__ == "__main__":
    indexar_documentos()