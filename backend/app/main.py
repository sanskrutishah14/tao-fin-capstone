# backend/app/main.py

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.table_parser import extract_table_chunks
from app.routes.sec import router as sec_router
from app.routes.analyze import router as analyze_router
from app.routes.market import router as market_router
from app.routes.companies import router as companies_router

from app.db.database import SessionLocal, init_db
from app.db.models import SECFiling

from app.services.chunker import split_into_chunks
from app.services.embeddings import EmbeddingService
from app.services.ollama_client import OllamaClient
from app.services.rag_pipeline import RAGPipeline
from app.services.tao_pipeline import TAOPipeline
from app.services.vector_store import FAISSVectorStore
from app.services.bm25_index import BM25Index
from app.services.reranker import Reranker
from app.services.hybrid_retriever import HybridRetriever


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
app.include_router(companies_router)


@app.on_event("startup")
def build_rag_index():
    """
    1. Create DB tables if they don't exist yet.
    2. Load every SECFiling row that has a local_text_path (i.e. has
       actually been downloaded+parsed via POST /api/companies/{ticker}/sync),
       chunk + embed them, and build FAISS + BM25 indices over the
       combined set. The resulting hybrid retriever + TAO pipeline are
       kept in app.state for the lifetime of the process.

    If nothing has been synced yet, /api/analyze will 503 -- run
    POST /api/companies/{TICKER}/sync at least once first, then
    restart the server (there's no hot-reload of the index yet).
    """

    init_db()

    db = SessionLocal()
    try:
        filings = (
            db.query(SECFiling)
            .filter(SECFiling.local_text_path.isnot(None))
            .all()
        )
        # Snapshot what we need before closing the session.
        filing_records = [
            {
                "text_path": Path(f.local_text_path),
                "html_path": Path(f.local_html_path) if f.local_html_path else None,
                "ticker": f.company.ticker,
                "form": f.form,
            }
            for f in filings
        ]
    finally:
        db.close()

    if not filing_records:
        print(
            "[startup] No synced filings found in the database -- "
            "POST /api/companies/{ticker}/sync at least one company, "
            "then restart. /api/analyze will 503 until then."
        )
        app.state.tao_pipeline = None
        return

    print(f"[startup] Indexing {len(filing_records)} filing(s) from the database...")

    embedding_service = EmbeddingService()

    all_chunks = []
    for record in filing_records:
        if not record["text_path"].exists():
            print(f"[startup] Skipping missing file: {record['text_path']}")
            continue

        text = record["text_path"].read_text(encoding="utf-8")
        all_chunks.extend(
            split_into_chunks(
                text=text,
                chunk_size=1200,
                overlap=200,
                metadata={
                    "source": "SEC EDGAR",
                    "company": record["ticker"],
                    "form": record["form"],
                    "filing": record["text_path"].name,
                    "type": "narrative",
                },
            )
        )

        if record["html_path"] and record["html_path"].exists():
            table_chunks = extract_table_chunks(
                str(record["html_path"]),
                metadata={
                    "source": "SEC EDGAR",
                    "company": record["ticker"],
                    "form": record["form"],
                    "filing": record["text_path"].name,
                },
            )
            all_chunks.extend(table_chunks)
            print(f"[startup] Extracted {len(table_chunks)} table chunk(s) from {record['html_path'].name}")

    embeddings = embedding_service.embed_documents(
        [c["text"] for c in all_chunks]
    )

    vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
    vector_store.add(embeddings, all_chunks)

    bm25_index = BM25Index()
    bm25_index.build(all_chunks)

    reranker = Reranker()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_service=embedding_service,
        reranker=reranker,
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        ollama_client=OllamaClient(),
    )

    app.state.tao_pipeline = TAOPipeline(rag_pipeline)

    print(
        f"[startup] Indexed {len(all_chunks)} chunks from {len(filing_records)} filing(s) "
        f"(FAISS + BM25, cross-encoder rerank enabled)."
    )


@app.get("/")
def root():
    return {
        "message": "TAO-Fin API is running"
    }