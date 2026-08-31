# backend/app/services/verifier.py

"""
Verifier

Scores a candidate answer against its retrieved evidence on four
axes (matching the frontend mock: Numerical, Evidence, Logical,
Financial consistency). This is deliberately heuristic + the
Calculator rather than another LLM call for the base checks --
that's what keeps "verification calls" cheap in the TAO loop.
"""

import re
from typing import Dict, List, Optional

from app.services.calculator import (
    extract_numbers,
    numbers_approximately_match,
)

_CITATION_PATTERN = re.compile(r"evidence\s+\d+", re.IGNORECASE)


REFUSAL_PHRASE = "insufficient evidence"


class VerificationResult:
    def __init__(
        self,
        numerical_ok: bool,
        evidence_ok: bool,
        logical_ok: bool,
        financial_ok: bool,
        confidence: float,
        failed_checks: List[str],
        details: Dict,
    ):
        self.numerical_ok = numerical_ok
        self.evidence_ok = evidence_ok
        self.logical_ok = logical_ok
        self.financial_ok = financial_ok
        self.confidence = confidence
        self.failed_checks = failed_checks
        self.details = details

    def to_dict(self) -> Dict:
        return {
            "checks": {
                "numerical": self.numerical_ok,
                "evidence": self.evidence_ok,
                "logical": self.logical_ok,
                "financial_consistency": self.financial_ok,
            },
            "confidence": round(self.confidence, 4),
            "failed_checks": self.failed_checks,
            "details": self.details,
        }


def _check_numerical(answer: str, evidence_text: str) -> bool:
    """Every number the answer states should be traceable to the evidence."""

    # Strip citation markers like "(Evidence 1)" first -- otherwise the
    # citation index itself gets picked up as an unverifiable "number".
    cleaned_answer = _CITATION_PATTERN.sub("", answer)
    answer_numbers = extract_numbers(cleaned_answer)

    if not answer_numbers:
        # No numeric claims made -- nothing to contradict.
        return True

    evidence_numbers = extract_numbers(evidence_text)

    if not evidence_numbers:
        return False

    for n in answer_numbers:
        if not any(numbers_approximately_match(n, e) for e in evidence_numbers):
            return False

    return True


def _check_evidence(answer: str, results: List[Dict]) -> bool:
    """Did the answer actually cite/ground itself in the retrieved evidence?"""

    lower_answer = answer.lower()

    if REFUSAL_PHRASE in lower_answer:
        # A refusal is only "grounded" if the evidence genuinely doesn't
        # cover the question -- we treat this as evidence-ok=False so the
        # TAO controller considers expanding retrieval.
        return False

    cited_evidence_number = any(
        f"evidence {i}" in lower_answer for i in range(1, len(results) + 1)
    )

    # Lexical overlap as a fallback signal if the model didn't literally
    # write "Evidence N".
    evidence_tokens = set()
    for r in results:
        evidence_tokens |= set(r["document"].get("text", "").lower().split())

    answer_tokens = set(lower_answer.split())
    overlap = len(answer_tokens & evidence_tokens) / max(len(answer_tokens), 1)

    return cited_evidence_number or overlap >= 0.15


def _check_logical(answer: str) -> bool:
    """Cheap heuristic: non-empty, not self-contradicting hedges."""

    if not answer or not answer.strip():
        return False

    lower = answer.lower()

    contradiction_markers = [
        ("insufficient evidence" in lower and any(ch.isdigit() for ch in answer)),
    ]

    return not any(contradiction_markers)


def _check_financial_consistency(
    answer: str,
    task_type: str,
    requires_calculation: bool,
) -> bool:
    """
    If the question required a derived metric (margin, growth rate, etc.),
    make sure the answer actually contains a number -- a purely narrative
    answer to a calculation question is a financial-consistency failure.
    """

    if not requires_calculation:
        return True

    return bool(extract_numbers(answer))


def verify(
    answer: str,
    results: List[Dict],
    task_type: str = "lookup",
    requires_calculation: bool = False,
) -> VerificationResult:

    evidence_text = "\n".join(
        r["document"].get("text", "") for r in results
    )

    numerical_ok = _check_numerical(answer, evidence_text)
    evidence_ok = _check_evidence(answer, results)
    logical_ok = _check_logical(answer)
    financial_ok = _check_financial_consistency(answer, task_type, requires_calculation)

    checks = [numerical_ok, evidence_ok, logical_ok, financial_ok]
    confidence = sum(checks) / len(checks)

    failed = []
    if not numerical_ok:
        failed.append("numerical")
    if not evidence_ok:
        failed.append("evidence")
    if not logical_ok:
        failed.append("logical")
    if not financial_ok:
        failed.append("financial_consistency")

    return VerificationResult(
        numerical_ok=numerical_ok,
        evidence_ok=evidence_ok,
        logical_ok=logical_ok,
        financial_ok=financial_ok,
        confidence=confidence,
        failed_checks=failed,
        details={
            "answer_numbers": extract_numbers(answer),
        },
    )