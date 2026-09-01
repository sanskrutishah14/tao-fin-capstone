import re

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

    # ============================================================
    # QUERY TERM EXTRACTION
    # ============================================================

    def _extract_query_terms(
        self,
        question: str
    ):
        """
        Extract meaningful terms from the user's question.

        No financial metrics, companies, or domains
        are hardcoded here.
        """

        stop_words = {
            "what",
            "was",
            "were",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "in",
            "for",
            "to",
            "from",
            "on",
            "and",
            "or",
            "how",
            "much",
            "many",
            "did",
            "does",
            "do",
            "has",
            "have",
            "had",
            "this",
            "that",
            "these",
            "those",
            "their",
            "its",
            "it",
            "company",
            "tell",
            "me",
            "give",
            "show",
            "please",
            "can",
            "you"
        }

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower()
        )

        terms = set()

        for word in words:

            if word in stop_words:
                continue

            # Keep years.
            if re.fullmatch(
                r"20\d{2}",
                word
            ):
                terms.add(word)
                continue

            # Ignore very short words.
            if len(word) < 3:
                continue

            terms.add(word)

        return terms

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5
    ):
        """
        Hybrid retrieval:

        1. Semantic similarity using FAISS.
        2. Generic lexical matching.
        3. Generic year matching.
        4. Re-ranking.

        No financial concepts are hardcoded.
        """

        query_embedding = (
            self.embedding_service.embed_query(
                question
            )
        )

        # Retrieve a larger candidate pool first.
        candidate_k = max(
            top_k * 5,
            20
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=candidate_k
        )

        if not results:
            return []

        query_terms = self._extract_query_terms(
            question
        )

        # Extract years separately.
        query_years = set(
            re.findall(
                r"\b20\d{2}\b",
                question
            )
        )

        reranked = []

        for result in results:

            document = result["document"]

            text = document.get(
                "text",
                ""
            )

            text_lower = text.lower()

            semantic_score = result["score"]

            # ----------------------------------------------------
            # Generic lexical matching
            # ----------------------------------------------------

            matched_terms = []

            for term in query_terms:

                pattern = (
                    r"\b"
                    + re.escape(term.lower())
                    + r"\b"
                )

                if re.search(
                    pattern,
                    text_lower
                ):
                    matched_terms.append(term)

            if query_terms:

                keyword_score = (
                    len(matched_terms)
                    / len(query_terms)
                )

            else:

                keyword_score = 0.0

            # ----------------------------------------------------
            # Generic year matching
            # ----------------------------------------------------

            matched_years = []

            for year in query_years:

                if re.search(
                    r"\b"
                    + re.escape(year)
                    + r"\b",
                    text
                ):
                    matched_years.append(year)

            if query_years:

                year_score = (
                    len(matched_years)
                    / len(query_years)
                )

            else:

                year_score = 0.0

            # ----------------------------------------------------
            # Combined score
            # ----------------------------------------------------

            final_score = (
                (semantic_score * 0.70)
                + (keyword_score * 0.20)
                + (year_score * 0.10)
            )

            reranked.append({
                "score": float(
                    final_score
                ),
                "semantic_score": float(
                    semantic_score
                ),
                "keyword_score": float(
                    keyword_score
                ),
                "year_score": float(
                    year_score
                ),
                "matched_terms": matched_terms,
                "matched_years": matched_years,
                "document": document
            })

        # Highest score first.
        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked[:top_k]

    # ============================================================
    # PROMPT BUILDING
    # ============================================================

    def build_prompt(
            self,
            question: str,
            results,
            revision_feedback: dict | None = None
        ):
            """
            Build a strict evidence-grounded prompt for Ollama.
            The model must return only a concise answer and evidence reference.
            """

            evidence_blocks = []

            for i, result in enumerate(results, start=1):

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

        SOURCE: {metadata.get("source", "Unknown")}
        COMPANY: {metadata.get("company", "Unknown")}
        FILING: {metadata.get("filing", "Unknown")}
        FORM: {metadata.get("form", "Unknown")}

        TEXT:
        {document.get("text", "")}
        """
                )

            evidence = "\n".join(evidence_blocks)

            prompt = f"""
        You are a financial question-answering system.

        Answer the user's question using ONLY the provided evidence.

        IMPORTANT:

        - Answer the EXACT question asked.
        - Identify the company, metric, and year requested.
        - Use the exact value from the evidence when available.
        - Do not use outside knowledge.
        - Do not invent or calculate values unless the evidence requires a simple calculation.
        - If multiple years appear, use only the requested year.
        - Ignore unrelated information.
        - Preserve the units from the evidence.
        - If the answer is explicitly present in the evidence, use it.
        - If the evidence does not contain enough information, answer:
        Insufficient evidence.

        DO NOT:
        - explain your reasoning
        - describe what the question is asking
        - describe the company
        - repeat the question
        - discuss the evidence
        - provide a summary
        - mention your internal reasoning
        - answer a related question
        - add financial information that was not requested

        Your response MUST contain only:

        Answer:
        <one or two concise sentences directly answering the question>

        Evidence:
        <EVIDENCE number(s) supporting the answer>

        USER QUESTION:
        {question}

        PROVIDED EVIDENCE:

        {evidence}
        """

            if revision_feedback:
                prompt += f"""

        The previous answer was incorrect.

        Previous answer:
        {revision_feedback.get("previous_answer", "")}

        Failed checks:
        {", ".join(revision_feedback.get("failed_checks", []))}

        Generate a corrected answer.

        Again, output ONLY:

        Answer:
        <direct answer>

        Evidence:
        <EVIDENCE number(s)>
        """

            return prompt

    # ============================================================
    # COMPLETE RAG PIPELINE
    # ============================================================

    def answer(
        self,
        question: str,
        top_k: int = 3
    ):
        """
        Complete RAG pipeline:

        User question
            ↓
        Query embedding
            ↓
        FAISS semantic retrieval
            ↓
        Lexical/year re-ranking
            ↓
        Top-K evidence
            ↓
        Ollama LLM
            ↓
        Grounded answer
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
