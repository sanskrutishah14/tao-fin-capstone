
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
        results,
        revision_feedback: dict | None = None
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

        Your task is to answer the USER QUESTION using ONLY the
        PROVIDED EVIDENCE.

        ========================
        STRICT RULES
        ========================

        1. Answer ONLY the exact question asked by the user.

        2. Do NOT answer a different question, even if the evidence
        contains information about other financial topics.

        3. Use ONLY information explicitly contained in the provided
        evidence.

        4. Do NOT use pretrained knowledge or outside information.

        5. Do NOT make assumptions, estimates, or invent numbers.

        6. Identify the exact:
        - company/entity
        - financial metric
        - financial year or period
        requested by the user.

        7. If the requested value appears directly in the evidence,
        use that value exactly.

        8. Preserve the units stated in the evidence.
        For example, if the evidence states "(in millions)",
        report the value in millions.

        9. If multiple years are present, use ONLY the year requested
        by the user.

        10. Ignore unrelated information in the evidence.

        11. If the requested information is explicitly present in
            the evidence, NEVER respond with "Insufficient evidence."

        12. Respond with "Insufficient evidence." ONLY when the
            provided evidence genuinely does not contain enough
            information to answer the user's exact question.

        13. Keep the answer concise.

        14. Include the evidence number that directly supports the
            answer.

        15. Do not explain your reasoning or analysis.

        ========================
        REQUIRED OUTPUT FORMAT
        ========================

        Answer:
        <direct answer to the user's question>

        Evidence:
        <EVIDENCE number(s) supporting the answer>

        ========================
        USER QUESTION
        ========================

        {question}

        ========================
        PROVIDED EVIDENCE
        ========================

        {evidence}

        ========================
        INTERNAL CHECK
        ========================

        Before answering, internally determine:

        - What company/entity is being asked about?
        - What metric is being requested?
        - What financial year/period is requested?
        - Which evidence contains that metric?
        - Does the evidence explicitly contain the requested value?
        - Are the units correct?
        - Does the answer address ONLY the user's question?

        Do NOT output this reasoning.

        """

        if revision_feedback:
            prompt += f"""
        ========================
        PREVIOUS ANSWER FEEDBACK
        ========================

        Your previous answer was:

        "{revision_feedback.get("previous_answer", "")}"

        The previous answer failed these checks:

        {", ".join(revision_feedback.get("failed_checks", []))}

        Correct the answer.

        You MUST:
        - answer only the original user question
        - use only the provided evidence
        - use the correct company/entity
        - use the correct metric
        - use the correct financial year
        - use the exact value from the evidence when available
        - preserve the evidence's units
        - include the supporting evidence number
        - avoid unrelated information
        - avoid saying "Insufficient evidence" when the answer is
        explicitly present in the evidence

        Do NOT output your reasoning.
        """

        prompt += """
        ========================
        FINAL ANSWER
        ========================
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
