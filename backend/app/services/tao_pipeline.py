# backend/app/services/tao_pipeline.py

"""
TAO Pipeline

The orchestrator matching the architecture diagram:

  question -> query analyzer -> retrieval -> reasoning LLM
           -> verifier -> TAO controller -> {stop, revise, retrieve}
           -> verified answer + evaluation metrics
"""

import time
from typing import Dict

from app.services.query_analyzer import analyze_query
from app.services.rag_pipeline import RAGPipeline
from app.services.tao_controller import Action, TAOState, decide
from app.services.verifier import verify

MAX_TOTAL_ITERATIONS = 5  # hard safety cap regardless of budgets


class TAOPipeline:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline

    def run(self, question: str) -> Dict:
        start = time.time()

        analysis = analyze_query(question)

        state = TAOState(
            max_reasoning_iterations=analysis["max_reasoning_iterations"],
            max_retrieval_expansions=analysis["max_retrieval_expansions"],
        )

        top_k = analysis["top_k"]
        results = self.rag_pipeline.retrieve(question, top_k=top_k)

        if not results:
            return {
                "answer": "Insufficient evidence.",
                "evidence": [],
                "verification": None,
                "query_analysis": analysis,
                "tao": state.to_dict(),
                "latency_seconds": round(time.time() - start, 2),
            }

        prompt = self.rag_pipeline.build_prompt(question, results)
        answer = self.rag_pipeline.ollama_client.generate(prompt)
        state.reasoning_iterations += 1

        verification = verify(
            answer,
            results,
            task_type=analysis["task_type"],
            requires_calculation=analysis["requires_calculation"],
        )
        state.verification_calls += 1

        iterations = 1

        while iterations < MAX_TOTAL_ITERATIONS:
            action = decide(verification, state)

            if action == Action.STOP:
                break

            if action == Action.RETRIEVE:
                top_k += 2
                results = self.rag_pipeline.retrieve(question, top_k=top_k)
                state.retrieval_expansions += 1

                prompt = self.rag_pipeline.build_prompt(question, results)
                answer = self.rag_pipeline.ollama_client.generate(prompt)
                state.reasoning_iterations += 1

            elif action == Action.REVISE:
                prompt = self.rag_pipeline.build_prompt(
                    question,
                    results,
                    revision_feedback={
                        "previous_answer": answer,
                        "failed_checks": verification.failed_checks,
                    },
                )
                answer = self.rag_pipeline.ollama_client.generate(prompt)
                state.reasoning_iterations += 1

            verification = verify(
                answer,
                results,
                task_type=analysis["task_type"],
                requires_calculation=analysis["requires_calculation"],
            )
            state.verification_calls += 1

            iterations += 1

        return {
            "answer": answer,
            "evidence": [
                {
                    "score": r["score"],
                    "source": r["document"].get("metadata", {}).get("source", "Unknown"),
                    "filing": r["document"].get("metadata", {}).get("filing", "Unknown"),
                    "text_preview": r["document"].get("text", "")[:300],
                }
                for r in results
            ],
            "verification": verification.to_dict(),
            "query_analysis": analysis,
            "tao": state.to_dict(),
            "latency_seconds": round(time.time() - start, 2),
        }