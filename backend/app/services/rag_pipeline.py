
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.services.ollama_client import OllamaClient


class RAGPipeline:

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_service: EmbeddingService,
        ollama_client: OllamaClient
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.ollama_client = ollama_client

    def retrieve(
        self,
        question: str,
        top_k: int = 2
    ):
        """
        Convert the user's question into an embedding
        and retrieve the most relevant SEC evidence
        from FAISS.
        """

        query_embedding = (
            self.embedding_service.embed_query(
                question
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        return results

    def build_prompt(
        self,
        question: str,
        results
    ):
        """
        Build a strict evidence-grounded prompt
        for the Ollama LLM.
        """

        evidence_blocks = []

        for i, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            metadata = document.get(
                "metadata",
                {}
            )

            evidence_blocks.append(
                f"""
====================
EVIDENCE {i}
====================

SOURCE:
{metadata.get("source", "Unknown")}

COMPANY:
{metadata.get("company", "Unknown")}

FILING:
{metadata.get("filing", "Unknown")}

FORM:
{metadata.get("form", "Unknown")}

EVIDENCE TEXT:
{document.get("text", "")}
"""
            )

        evidence = "\n".join(
            evidence_blocks
        )

        prompt = f"""
You are a financial question-answering system.

Your job is to answer the USER QUESTION using
ONLY the provided EVIDENCE.

STRICT RULES:

1. Answer ONLY the exact question asked.

2. Do NOT answer a different question, even if
   the evidence contains information about other
   financial topics.

3. Use ONLY information contained in the evidence.

4. Do NOT use your pretrained knowledge.

5. Do NOT make assumptions or invent numbers.

6. Identify the exact entity, metric, and financial
   year requested by the user.

7. If the requested value appears directly in the
   evidence, use that value exactly.

8. Preserve the original units from the evidence.
   For example, if the evidence says "in millions",
   report the value in millions.

9. If the evidence contains multiple years, use ONLY
   the year requested by the user.

10. Ignore unrelated information in the evidence.

11. If the evidence does not contain enough information
    to answer the question, respond exactly:

    Insufficient evidence.

12. Keep the final answer concise.

13. Mention the evidence number supporting your answer.

USER QUESTION:
{question}

PROVIDED EVIDENCE:
{evidence}

REASONING CHECK:

Before producing the final answer, determine:

- What company is being asked about?
- What financial metric is being requested?
- What financial year is requested?
- Which evidence contains the requested information?
- Is the requested value explicitly present?

Do not output this reasoning.

FINAL ANSWER:
"""

        return prompt

    def answer(
        self,
        question: str,
        top_k: int = 2
    ):
        """
        Complete RAG pipeline:

        Question
            ↓
        FAISS retrieval
            ↓
        Evidence
            ↓
        Prompt
            ↓
        Ollama LLM
            ↓
        Answer
        """

        results = self.retrieve(
            question,
            top_k=top_k
        )

        if not results:

            return {
                "answer": "Insufficient evidence.",
                "evidence": []
            }

        prompt = self.build_prompt(
            question,
            results
        )

        answer = self.ollama_client.generate(
            prompt
        )

        return {
            "answer": answer,
            "evidence": results
        }
