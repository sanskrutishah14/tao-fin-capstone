from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.sec import router as sec_router


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


@app.get("/")
def root():
    return {
        "message": "TAO-Fin API is running"
    }