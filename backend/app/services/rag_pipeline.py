
from app.services.hybrid_retriever import HybridRetriever
from app.services.ollama_client import OllamaClient

_CALCULATION_GUIDANCE = """
        16. This question requires a CALCULATED metric (margin, growth
            rate, ratio, etc.), not a single number that's printed
            verbatim in the filing. If the exact metric is not stated
            directly, but the underlying numbers ARE present in the
            evidence, COMPUTE it yourself using these formulas:

            - Operating margin = (Operating Income / Total Net Sales) x 100
            - Gross margin     = (Gross Profit / Total Net Sales) x 100
            - Growth rate      = ((New Value - Old Value) / Old Value) x 100

            State the computed number explicitly in your Answer (e.g.
            "Operating margin was 31.2%"). Do NOT just say the metric
            "is shown in the evidence" without stating the value.
            Only use numbers that literally appear in the evidence as
            inputs to the formula -- do not invent the inputs.
"""


class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRetriever,
        ollama_client: OllamaClient
    ):
        self.retriever = retriever
        self.ollama_client = ollama_client

    def retrieve(
        self,
        question: str,
        top_k: int = 5
    ):
        """
        Hybrid retrieval: FAISS (semantic) + BM25 (keyword), fused via
        Reciprocal Rank Fusion, then reranked by a cross-encoder to
        produce the final evidence set.
        """

        return self.retriever.retrieve(
            question,
            top_k=top_k
        )

    def build_prompt(
        self,
        question: str,
        results,
        revision_feedback: dict | None = None,
        requires_calculation: bool = False,
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
        {_CALCULATION_GUIDANCE if requires_calculation else ""}
        ========================
        REQUIRED OUTPUT FORMAT
        ========================

        Output ONLY the block below. Nothing before it, nothing after
        it. Do NOT include your internal check, your reasoning, or
        any text other than these two lines:

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
        INTERNAL CHECK (silent -- do not print this section or its answers)
        ========================

        Before answering, internally determine:

        - What company/entity is being asked about?
        - What metric is being requested?
        - What financial year/period is requested?
        - Which evidence contains that metric (or its inputs)?
        - Does the evidence explicitly contain the requested value,
          or the numbers needed to compute it?
        - Are the units correct?
        - Does the answer address ONLY the user's question?

        Do NOT output any of this. Output ONLY the "Answer:" /
        "Evidence:" block above.

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
        - use the exact value from the evidence when available, or
        compute it per the CALCULATION GUIDANCE above if needed
        - preserve the evidence's units
        - include the supporting evidence number
        - avoid unrelated information
        - avoid saying "Insufficient evidence" when the answer is
        explicitly present in the evidence
        - output ONLY the "Answer:" / "Evidence:" block, nothing else

        Do NOT output your reasoning.
        """

        prompt += """
        ========================
        FINAL ANSWER
        ========================
        """


        return prompt

    @staticmethod
    def _clean_answer(raw_text: str) -> str:
        """
        Small local models sometimes ignore "don't show your reasoning"
        and print their internal-check notes before the real answer.
        If the model DID follow instructions and emitted "Answer:",
        cut everything before that marker so leaked reasoning never
        ends up in the final answer field.
        """
        marker = "answer:"
        lower = raw_text.lower()
        idx = lower.find(marker)

        if idx == -1:
            # Model didn't use the marker at all -- return as-is rather
            # than silently dropping content.
            return raw_text.strip()

        return raw_text[idx:].strip()

    def answer(
        self,
        question: str,
        top_k: int = 5,
        requires_calculation: bool = False,
    ):
        """
        Complete RAG pipeline:

        Question
            ↓
        Hybrid retrieval (FAISS + BM25 + RRF + rerank)
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
            results,
            requires_calculation=requires_calculation,
        )

        raw_answer = self.ollama_client.generate(
            prompt
        )

        return {
            "answer": self._clean_answer(raw_answer),
            "evidence": results
        }