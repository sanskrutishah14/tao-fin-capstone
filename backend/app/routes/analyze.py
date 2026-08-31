from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(
    prefix="/api/analyze",
    tags=["Analyze"]
)


class AnalyzeRequest(BaseModel):
    question: str


@router.post("")
def analyze(request: Request, body: AnalyzeRequest):
    tao_pipeline = getattr(request.app.state, "tao_pipeline", None)

    if tao_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG index is not ready yet. Make sure data/sec/*.txt "
                "files exist and the server finished its startup indexing."
            ),
        )

    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        return tao_pipeline.run(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))