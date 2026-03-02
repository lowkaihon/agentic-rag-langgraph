"""HHEM hallucination detector with pluggable backends (local model or Vectara API).

Backend selection via HHEM_BACKEND environment variable:
- HHEM_BACKEND=local (default): Local HuggingFace model, works forever, no API keys needed
- HHEM_BACKEND=vectara: Vectara managed API (HHEM-2.3), faster, requires API credentials
"""

from typing import TypedDict, List, Protocol
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
import os
import re
import logging

logger = logging.getLogger(__name__)


class ClaimDecomposition(TypedDict):
    """Structured output schema for claim extraction"""
    claims: List[str]
    reasoning: str


class HHEMBackend(Protocol):
    """Protocol for HHEM backend implementations."""

    def evaluate_claim(self, claim: str, contexts: List[str]) -> float:
        """Evaluate claim against contexts, return consistency score 0-1."""
        ...


class LocalHHEMBackend:
    """Local HHEM-2.1-Open backend using HuggingFace transformers.

    Works forever without API keys. Uses batch inference for efficiency.
    """

    display_name = "HHEM-2.1-Open (local)"

    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model"):
        """Initialize local HHEM model with batch inference."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        import ctypes

        # Suppress HuggingFace transformers warnings about custom model config (HHEMv2Config)
        logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

        self.hhem_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.hhem_model.eval()

        # Tokenizer for input truncation (HHEM max sequence length is 512 tokens)
        # HHEM uses custom config, but it's based on FLAN-T5-Base
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.torch = torch
        self.ctypes = ctypes

    def _truncate_pair(self, claim: str, context: str) -> tuple:
        """Truncate context to fit within HHEM's 512 token limit."""
        MAX_TOTAL_TOKENS = 500

        claim_tokens = self.tokenizer.encode(claim, add_special_tokens=False)
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        max_context_tokens = MAX_TOTAL_TOKENS - len(claim_tokens)

        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)

        return (context, claim)

    def evaluate_claim(self, claim: str, contexts: List[str]) -> float:
        """Evaluate claim against all contexts, return max consistency score.

        Uses batch inference for efficiency.
        """
        if not contexts:
            return 0.5

        # Build all (context, claim) pairs for batch inference
        all_pairs = [self._truncate_pair(claim, ctx) for ctx in contexts]

        # Single batch inference call
        with self.torch.inference_mode():
            scores = self.hhem_model.predict(all_pairs)

        # Return max score across all contexts
        max_score = max(float(s) for s in scores)

        # Release memory to OS (glibc malloc_trim) - fixes Azure hang on 2nd evaluation
        try:
            libc = self.ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except OSError:
            pass  # Windows/non-glibc systems

        return max_score


class VectaraHHEMBackend:
    """Vectara managed HHEM-2.3 API backend.

    Faster, zero memory footprint, but requires API credentials and has usage limits.
    """

    display_name = "HHEM-2.3 (API)"
    API_URL = "https://api.vectara.io/v2/evaluate_factual_consistency"

    def __init__(self, api_key: str = None, customer_id: str = None):
        """Initialize with Vectara API credentials."""
        import requests

        self.requests = requests
        self.api_key = api_key or os.getenv("VECTARA_API_KEY")
        self.customer_id = customer_id or os.getenv("VECTARA_CUSTOMER_ID")

        if not self.api_key:
            raise ValueError("VECTARA_API_KEY environment variable not set")
        if not self.customer_id:
            raise ValueError("VECTARA_CUSTOMER_ID environment variable not set")

        self.headers = {
            "Content-Type": "application/json",
            "customer-id": self.customer_id,
            "x-api-key": self.api_key
        }

    def evaluate_claim(self, claim: str, contexts: List[str]) -> float:
        """Evaluate claim against all contexts using Vectara API.

        API accepts all contexts in one call (efficient).
        """
        if not contexts:
            return 0.5

        payload = {
            "generated_text": claim,
            "source_texts": contexts
        }

        try:
            response = self.requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("score", 0.5)
        except self.requests.exceptions.RequestException as e:
            logger.error(f"Vectara API error: {e}")
            return 0.0  # Fail-safe: treat as unsupported to trigger regeneration


def create_hhem_backend() -> HHEMBackend:
    """Factory function to create HHEM backend based on HHEM_BACKEND env var.

    Returns:
        LocalHHEMBackend if HHEM_BACKEND=local (default)
        VectaraHHEMBackend if HHEM_BACKEND=vectara
    """
    backend_type = os.getenv("HHEM_BACKEND", "local").lower()

    if backend_type == "vectara":
        backend = VectaraHHEMBackend()
        print(f"Using Vectara HHEM API backend (HHEM-2.3)")
        return backend
    else:
        backend = LocalHHEMBackend()
        print(f"Using local HHEM backend (HHEM-2.1-Open)")
        return backend


class HHEMHallucinationDetector:
    """HHEM-based hallucination detector with pluggable backends.

    Score 0=hallucination, 1=consistent.

    Backend selection via HHEM_BACKEND environment variable:
    - local (default): HuggingFace model, works forever
    - vectara: Managed API, faster but requires credentials
    """

    def __init__(
        self,
        llm_model: str = None,
        entailment_threshold: float = 0.5,
        backend: HHEMBackend = None
    ):
        """Initialize with HHEM backend and LLM for claim decomposition.

        Args:
            llm_model: Model for claim decomposition (default from model_config)
            entailment_threshold: Score threshold for claim support (default 0.5)
            backend: HHEM backend instance (default from create_hhem_backend())
        """
        self.backend = backend or create_hhem_backend()

        spec = get_model_for_task("hhem_claim_decomposition")
        llm_model = llm_model or spec.name

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=spec.temperature,
            max_tokens=1024,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity
        )
        self.structured_llm = self.llm.with_structured_output(ClaimDecomposition)
        self.entailment_threshold = entailment_threshold

    @property
    def backend_display_name(self) -> str:
        """Get user-friendly backend name for logging."""
        return getattr(self.backend, 'display_name', type(self.backend).__name__)

    def decompose_into_claims(self, answer: str) -> List[str]:
        """Decompose answer into atomic factual claims using LLM."""
        from advanced_agentic_rag_langgraph.prompts import get_prompt

        decomposition_prompt = get_prompt("hhem_claim_decomposition", answer=answer)

        try:
            result = self.structured_llm.invoke([HumanMessage(content=decomposition_prompt)])
            return result["claims"]
        except Exception as e:
            print(f"Warning: Claim decomposition failed: {e}. Using fallback.")
            return [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]

    def verify_claim_entailment(self, claim: str, context: str) -> dict:
        """Verify single claim against context using HHEM backend."""
        consistency_score = self.backend.evaluate_claim(claim, [context])
        supported = consistency_score >= self.entailment_threshold

        return {
            "entailment_score": consistency_score,
            "consistency_score": consistency_score,
            "label": "supported" if supported else "unsupported",
            "supported": supported
        }

    def verify_groundedness(self, answer: str, chunks: List[str]) -> dict:
        """Verify answer groundedness via claim verification against all chunks.

        Both backends optimize multi-context evaluation:
        - Local: Batch inference (all claim-context pairs in one call)
        - Vectara: API accepts all contexts in one request per claim
        """
        if not answer or not chunks:
            return {
                "claims": [],
                "entailment_scores": [],
                "supported": [],
                "unsupported_claims": [],
                "groundedness_score": 1.0,
                "claim_details": [],
                "reasoning": "Empty answer or chunks"
            }

        claims = self.decompose_into_claims(answer)

        if not claims:
            return {
                "claims": [],
                "entailment_scores": [],
                "supported": [],
                "unsupported_claims": [],
                "groundedness_score": 1.0,
                "claim_details": [],
                "reasoning": "No claims extracted from answer"
            }

        claim_details = []
        entailment_scores = []
        supported_flags = []
        unsupported_claims = []

        for claim in claims:
            # Backend handles multi-context evaluation efficiently
            score = self.backend.evaluate_claim(claim, chunks)
            supported = score >= self.entailment_threshold

            claim_details.append({
                "claim": claim,
                "entailment_score": score,
                "label": "supported" if supported else "unsupported",
                "supported": supported,
                "best_chunk_idx": 0,  # Backend abstraction doesn't expose per-chunk scores
                "chunk_scores": [score]  # Aggregated score
            })

            entailment_scores.append(score)
            supported_flags.append(supported)

            if not supported:
                unsupported_claims.append(claim)

        total_claims = len(claims)
        supported_count = sum(supported_flags)
        groundedness_score = supported_count / total_claims if total_claims > 0 else 1.0
        reasoning = f"Verified {total_claims} claims against {len(chunks)} chunks: {supported_count} supported, {len(unsupported_claims)} unsupported"

        return {
            "claims": claims,
            "entailment_scores": entailment_scores,
            "supported": supported_flags,
            "unsupported_claims": unsupported_claims,
            "groundedness_score": groundedness_score,
            "claim_details": claim_details,
            "reasoning": reasoning
        }
