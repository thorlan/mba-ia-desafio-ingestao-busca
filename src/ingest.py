import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")


def ingest_pdf():
    # 1. Carrega o PDF - cada página vira um Document (page_content + metadata)
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 2. Quebra em chunks de 1000 caracteres com overlap de 150
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # 3. Modelo de embeddings do Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # 4. Conecta no PGVector e grava os chunks vetorizados
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,  # limpa a coleção antes de reingestar (idempotente)
    )
    # 5. Grava em lotes pequenos, com pausa entre eles, pra não estourar o
    #    limite de requisições por minuto (RPM) do free tier do Gemini
    BATCH_SIZE = 15
    PAUSE_SECONDS = 20

    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"Gravados {min(i + BATCH_SIZE, total)}/{total} chunks...")
        if i + BATCH_SIZE < total:
            time.sleep(PAUSE_SECONDS)

    print(f"Ingestão concluída: {total} chunks gravados no banco.")


if __name__ == "__main__":
    ingest_pdf()