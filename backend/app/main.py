# backend/app/main.py

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.table_parser import extract_table_chunks
from app.routes.sec import router as sec_router
from app.routes.analyze import router as analyze_router
from app.routes.market import router as market_router

from app.services.chunker import split_into_chunks
from app.services.embeddings import EmbeddingService
from app.services.ollama_client import OllamaClient
from app.services.rag_pipeline import RAGPipeline
from app.services.tao_pipeline import TAOPipeline
from app.services.vector_store import FAISSVectorStore


app = FastAPI(
    title="TAO-Fin API",
    description="Verifier-guided financial reasoning system",
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(sec_router)
app.include_router(analyze_router)
app.include_router(market_router)


@app.on_event("startup")
def build_rag_index():
    """
    Load every parsed filing under data/sec/*.txt, chunk + embed them
    once, and keep the resulting FAISS index + TAO pipeline in
    app.state for the lifetime of the process. Re-run the SEC
    download/parse tests to add more filings, then restart the server
    to re-index -- there's no incremental indexing yet.
    """

    sec_folder = Path("data/sec")
    text_files = sorted(sec_folder.glob("*.txt")) if sec_folder.exists() else []

    if not text_files:
        print("[startup] No data/sec/*.txt files found -- /api/analyze will 503.")
        app.state.tao_pipeline = None
        return

    print(f"[startup] Indexing {len(text_files)} filing(s)...")

    embedding_service = EmbeddingService()

    all_chunks = []
    for text_file in text_files:
        text = text_file.read_text(encoding="utf-8")
        all_chunks.extend(
            split_into_chunks(
                text=text,
                chunk_size=1200,
                overlap=200,
                metadata={
                    "source": "SEC EDGAR",
                    "filing": text_file.name,
                    "type": "narrative",
                },
            )
        )

        # If a matching .html exists next to the .txt, pull its tables
        # out as separate, unambiguous "label: year=value" chunks
        # instead of relying on flattened text.
        html_file = text_file.with_suffix(".html")
        if html_file.exists():
            table_chunks = extract_table_chunks(
                str(html_file),
                metadata={
                    "source": "SEC EDGAR",
                    "filing": text_file.name,
                },
            )
            all_chunks.extend(table_chunks)
            print(f"[startup] Extracted {len(table_chunks)} table chunk(s) from {html_file.name}")

    embeddings = embedding_service.embed_documents(
        [c["text"] for c in all_chunks]
    )

    vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
    vector_store.add(embeddings, all_chunks)

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        ollama_client=OllamaClient(),
    )

    app.state.tao_pipeline = TAOPipeline(rag_pipeline)

    print(f"[startup] Indexed {len(all_chunks)} chunks from {len(text_files)} filing(s).")


@app.get("/")
def root():
    return {
        "message": "TAO-Fin API is running"
    }