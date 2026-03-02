from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from advanced_agentic_rag_langgraph.retrieval import (
    expand_query,
    AdaptiveRetriever,
    LLMMetadataReRanker,
    SemanticRetriever,
)
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_ndcg
from advanced_agentic_rag_langgraph.validation import HHEMHallucinationDetector
from advanced_agentic_rag_langgraph.retrieval.query_optimization import optimize_query_for_strategy, rewrite_query
from advanced_agentic_rag_langgraph.prompts import get_prompt
from advanced_agentic_rag_langgraph.prompts.answer_generation import get_answer_generation_prompts
import re
import json


adaptive_retriever = None
conversational_rewriter = ConversationalRewriter()
strategy_selector = StrategySelector()
hhem_detector = HHEMHallucinationDetector()

# Print HHEM backend info at module load
print(f"\n{'='*60}")
print(f"HHEM HALLUCINATION DETECTOR INITIALIZED")
print(f"Backend: {hhem_detector.backend_display_name}")
backend_mode = "Offline, no API keys required" if "local" in hhem_detector.backend_display_name.lower() else "Online, faster inference (~200-500ms)"
print(f"Mode: {backend_mode}")
print(f"{'='*60}\n")


# ============ STRUCTURED OUTPUT SCHEMAS ============

class ExpansionDecision(TypedDict):
    """LLM structured output for query expansion decision."""
    decision: Literal["yes", "no"]
    reasoning: str


class RetrievalQualityEvaluation(TypedDict):
    """LLM structured output for retrieval quality assessment."""
    quality_score: float
    reasoning: str
    issues: list[str]
    keywords_to_inject: list[str]


class AnswerQualityEvaluation(TypedDict):
    """LLM structured output for answer quality assessment."""
    is_relevant: bool
    is_complete: bool
    is_accurate: bool
    confidence_score: float
    reasoning: str
    issues: list[str]


class RefusalCheck(TypedDict):
    """LLM structured output for refusal detection."""
    refused: bool
    reasoning: str


# ========== HELPER FUNCTIONS ==========

def _extract_conversation_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Extract conversation history as [{"user": str, "assistant": str}, ...] pairs."""
    if not messages or len(messages) < 2:
        return []

    conversation = []
    i = 0

    while i < len(messages) - 1:
        # Look for HumanMessage followed by AIMessage
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i+1], AIMessage):
            conversation.append({
                "user": messages[i].content,
                "assistant": messages[i+1].content
            })
            i += 2
        else:
            i += 1

    return conversation
    

def _should_skip_expansion_llm(query: str) -> bool:
    """Use LLM to determine if query expansion would improve retrieval."""
    spec = get_model_for_task("expansion_decision")
    expansion_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )

    prompt = f"""Should this query be expanded into multiple variations for better retrieval?

Query: "{query}"

EXPANSION IS BENEFICIAL FOR:
- Ambiguous queries that could be phrased multiple ways
- Complex questions where synonyms/related terms would help
- Queries where users might use different terminology
- Conceptual questions with multiple valid phrasings

SKIP EXPANSION FOR:
- Clear, specific queries that are already well-formed
- Simple factual lookups (definitions, direct questions)
- Queries with exact-match intent only (pure lookups)
- Procedural queries with specific steps (expansion adds noise)
- Queries that are already precise and unambiguous

IMPORTANT CONSIDERATIONS:
- Consider overall intent, not just query length or presence of quotes
- Quoted terms don't automatically mean skip - consider if variations help
- Example: "Compare 'X' and 'Y'" has quotes BUT expansion helps (synonyms for "compare")
- Example: "What is Z?" is simple BUT expansion might help (rephrasing)

Return your decision ('yes' or 'no') with brief reasoning."""

    try:
        structured_llm = expansion_llm.with_structured_output(ExpansionDecision)
        result = structured_llm.invoke(prompt)
        skip = (result["decision"] == "no")

        print(f"\n{'='*60}")
        print(f"EXPANSION DECISION")
        print(f"Query: {query}")
        print(f"LLM decision: {'SKIP expansion' if skip else 'EXPAND query'}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"{'='*60}\n")

        return skip
    except Exception as e:
        print(f"Warning: Expansion decision LLM failed: {e}, defaulting to expand")
        return False


# ============ CONVERSATIONAL QUERY REWRITING ============

def conversational_rewrite_node(state: dict) -> dict:
    """Rewrite query using conversation history to make it self-contained."""
    question = state.get("user_question", "")

    # Extract conversation from messages (LangGraph best practice)
    messages = state.get("messages", [])
    conversation_history = _extract_conversation_history(messages)

    rewritten_query, reasoning = conversational_rewriter.rewrite(
        question,
        conversation_history
    )

    if rewritten_query != question:
        print(f"\n{'='*60}")
        print(f"CONVERSATIONAL REWRITE")
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten_query}")
        print(f"Reasoning: {reasoning}")
        print(f"Conversation turns used: {len(conversation_history)}")
        print(f"State transition: user_question={question}, baseline_query={rewritten_query}")
        print(f"{'='*60}\n")

    return {
        "baseline_query": rewritten_query,
        "active_query": rewritten_query,
        "corpus_stats": get_corpus_stats(),
        "messages": [HumanMessage(content=question)],

        # Reset attempt counters and feedback
        "retrieval_attempts": 0,  # Reset counter for new user question
        "generation_attempts": 0,  # Reset for new user question (0 = no attempts yet)
        "retry_feedback": None,  # Clear feedback for new user question

        # Reset strategy switch state (prevents stale flags triggering incorrect revert on next turn)
        "strategy_changed": None,
        "previous_strategy": None,

        # Reset answer evaluation state (prevents state leakage across turns)
        "is_refusal": None,  # CRITICAL: Prevents routing logic treating new turn as refusal
        "is_answer_sufficient": None,
        "answer_quality_reasoning": None,
        "answer_quality_issues": None,

        # Reset groundedness/hallucination detection
        "groundedness_score": None,
        "has_hallucination": None,
        "unsupported_claims": None,

        # Reset output state
        "final_answer": None,
        "confidence_score": 0.0,
    }

# ============ QUERY OPTIMIZATION STAGE ============

def decide_retrieval_strategy_node(state: dict) -> dict:
    """Decide retrieval strategy (semantic/keyword/hybrid) based on query and corpus."""
    query = state["baseline_query"]
    corpus_stats = state.get("corpus_stats", {})

    strategy, confidence, reasoning = strategy_selector.select_strategy(
        query,
        corpus_stats
    )

    print(f"\n{'='*60}")
    print(f"STRATEGY SELECTION")
    print(f"Query: {query}")
    print(f"Selected: {strategy.upper()}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Reasoning: {reasoning}")
    print(f"Note: Query optimization will happen in query_expansion_node")
    print(f"{'='*60}\n")

    return {
        "retrieval_strategy": strategy,
        "messages": [AIMessage(content=f"Strategy: {strategy} (confidence: {confidence:.0%})")],
    }


def query_expansion_node(state: dict) -> dict:
    """Generate strategy-agnostic expansions, then optimize expansions[0] for strategy."""

    quality = state.get("retrieval_quality_score", 1.0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])
    current_strategy = state.get("retrieval_strategy", "hybrid")

    # Check for early strategy switch
    early_switch = (quality < 0.6 and
                    attempts == 1 and
                    ("off_topic" in issues or "wrong_domain" in issues))

    old_strategy = None
    strategy_updates = {}

    if early_switch:
        # Off-topic results indicate need for precision -> keyword search
        old_strategy = current_strategy
        next_strategy = "keyword" if current_strategy != "keyword" else "hybrid"

        print(f"\n{'='*60}")
        print(f"EARLY STRATEGY SWITCH")
        print(f"From: {current_strategy} to {next_strategy}")
        print(f"Reason: {', '.join(issues)}")
        print(f"{'='*60}\n")

        current_strategy = next_strategy  # Use new strategy for optimization

        strategy_updates = {
            "retrieval_strategy": next_strategy,
            "strategy_switch_reason": f"Early detection: {', '.join(issues)}",
            "strategy_changed": True,
            "previous_strategy": old_strategy,  # Store for revert validation
        }

    source_query = state.get("active_query", state["baseline_query"])

    # 1. Expand FIRST (strategy-agnostic variants for RRF diversity)
    if _should_skip_expansion_llm(source_query):
        expansions = [source_query]
    else:
        expansions = expand_query(source_query)
        print(f"\n{'='*60}")
        print(f"QUERY EXPANDED (Strategy-Agnostic)")
        print(f"Source query: {source_query}")
        print(f"Expansions: {expansions[1:]}")
        print(f"{'='*60}\n")

    # 2. Optimize for strategy
    optimized_query = optimize_query_for_strategy(
        query=source_query,
        strategy=current_strategy,
        old_strategy=old_strategy,  # Only set during early switch
        issues=issues if early_switch else []
    )

    # 3. Replace expansions[0] with optimized version
    expansions[0] = optimized_query

    result = {
        "retrieval_query": optimized_query,
        "query_expansions": expansions,
        **strategy_updates
    }
    return result

# ============ ADAPTIVE RETRIEVAL STAGE ============

def retrieve_with_expansion_node(state: dict) -> dict:
    """Retrieve documents using query expansions with RRF fusion and two-stage reranking."""

    global adaptive_retriever
    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    strategy = state.get("retrieval_strategy", "hybrid")

    doc_ranks = {}
    doc_objects = {}

    expansion_source = "retrieval_query" if state.get("retrieval_query") else "active_query"
    expansions_count = len(state.get("query_expansions", []))
    print(f"\n{'='*60}")
    print(f"RETRIEVAL EXECUTION START")
    print(f"Using {expansions_count} query expansion(s)")
    print(f"Expansions generated from: {expansion_source}")
    print(f"Retrieval strategy: {strategy}")
    print(f"{'='*60}\n")

    for query in state.get("query_expansions", []):
        docs = adaptive_retriever.retrieve_without_reranking(query, strategy=strategy)

        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
                doc_objects[doc_id] = doc
            doc_ranks[doc_id].append(rank)

    k = 60
    rrf_scores = {}
    for doc_id, ranks in doc_ranks.items():
        rrf_score = sum(1.0 / (rank + k) for rank in ranks)
        rrf_scores[doc_id] = rrf_score

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    unique_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    print(f"\n{'='*60}")
    print(f"RRF MULTI-QUERY RETRIEVAL")
    print(f"Query variants: {len(state['query_expansions'])}")
    print(f"Total retrievals: {sum(len(ranks) for ranks in doc_ranks.values())}")
    print(f"Unique docs after RRF: {len(unique_docs)}")

    # Show ALL chunk IDs with RRF scores (typically 16-22 chunks)
    print(f"\nAll {len(sorted_doc_ids)} chunk IDs (RRF scores):")
    for i, doc_id in enumerate(sorted_doc_ids, 1):
        print(f"  {i}. {doc_id} ({rrf_scores[doc_id]:.4f})")

    # Show ground truth tracking
    if ground_truth_doc_ids:
        found_chunks = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in sorted_doc_ids]
        missing_chunks = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in sorted_doc_ids]
        print(f"\nExpected chunks: {ground_truth_doc_ids}")
        print(f"Found: {found_chunks if found_chunks else '[]'} | Missing: {missing_chunks if missing_chunks else '[]'}")

    print(f"{'='*60}\n")

    reranking_input = unique_docs[:40]

    print(f"{'='*60}")
    print(f"TWO-STAGE RERANKING (After RRF)")
    print(f"Input: {len(reranking_input)} docs (from RRF top-40)")

    # Show chunk IDs going into reranking
    reranking_chunk_ids = [doc.metadata.get("id", "unknown") for doc in reranking_input]
    print(f"\nChunk IDs sent to reranking (top-40):")
    for i, chunk_id in enumerate(reranking_chunk_ids[:10], 1):
        print(f"  {i}. {chunk_id}")
    if len(reranking_chunk_ids) > 10:
        print(f"  ... and {len(reranking_chunk_ids) - 10} more")

    # Track ground truth in reranking input
    if ground_truth_doc_ids:
        found_in_reranking = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in reranking_chunk_ids]
        missing_in_reranking = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in reranking_chunk_ids]
        print(f"\nExpected chunks in reranking input:")
        print(f"Found: {found_in_reranking if found_in_reranking else '[]'} | Missing: {missing_in_reranking if missing_in_reranking else '[]'}")

    query_for_reranking = state.get('active_query', state['baseline_query'])

    query_source = "active_query" if state.get("active_query") else "baseline_query"
    query_type = "semantic, human-readable" if query_source == "active_query" else "conversational, self-contained"
    query_type_description = "semantic query" if query_source == "active_query" else "conversational query"
    print(f"\n{'='*60}")
    print(f"RERANKING QUERY SOURCE")
    print(f"Using: {query_source} ({query_type})")
    print(f"Query: {query_for_reranking}")
    print(f"Note: Reranking uses {query_type_description}, NOT algorithm-optimized retrieval_query")
    print(f"{'='*60}\n")

    ranked_results = adaptive_retriever.reranker.rank(
        query_for_reranking,
        reranking_input
    )

    unique_docs = [doc for doc, score in ranked_results]
    reranking_scores = [score for doc, score in ranked_results]

    print(f"\nOutput: {len(unique_docs)} docs after two-stage reranking")

    # Show final chunk IDs with reranking scores
    print(f"\nFinal chunk IDs (after two-stage reranking):")
    for i, (doc, score) in enumerate(zip(unique_docs, reranking_scores), 1):
        chunk_id = doc.metadata.get("id", "unknown")
        print(f"  {i}. {chunk_id} (score: {score:.4f})")

    # Track ground truth in final results
    if ground_truth_doc_ids:
        final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in unique_docs]
        found_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in final_chunk_ids]
        missing_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in final_chunk_ids]
        print(f"\nExpected chunks in final results:")
        print(f"Found: {found_in_final if found_in_final else '[]'} | Missing: {missing_in_final if missing_in_final else '[]'}")

    print(f"{'='*60}\n")

    docs_text = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in unique_docs
    ])

    spec = get_model_for_task("retrieval_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
    structured_quality_llm = quality_llm.with_structured_output(RetrievalQualityEvaluation)

    quality_prompt = get_prompt("retrieval_quality_eval", query=state.get('active_query', state['baseline_query']), docs_text=docs_text)

    try:
        evaluation = structured_quality_llm.invoke(quality_prompt)
        quality_score = evaluation["quality_score"] / 100
        quality_reasoning = evaluation["reasoning"]
        quality_issues = evaluation["issues"]
        keywords_to_inject = evaluation.get("keywords_to_inject", [])
    except Exception as e:
        print(f"Warning: Quality evaluation failed: {e}. Using neutral score.")
        quality_score = 0.5
        quality_reasoning = "Evaluation failed"
        quality_issues = []
        keywords_to_inject = []

    retrieval_metrics = {}
    ground_truth_doc_ids = state.get("ground_truth_doc_ids")
    relevance_grades = state.get("relevance_grades")

    if ground_truth_doc_ids:
        retrieval_metrics = calculate_retrieval_metrics(
            unique_docs,
            ground_truth_doc_ids,
            k=adaptive_retriever.k_final
        )

        if relevance_grades:
            retrieval_metrics["ndcg_at_k"] = calculate_ndcg(
                unique_docs,
                relevance_grades,
                k=adaptive_retriever.k_final
            )

        k = adaptive_retriever.k_final
        print(f"\n{'='*60}")
        print(f"RETRIEVAL METRICS (Golden Dataset Evaluation)")
        print(f"{'='*60}")
        print(f"Recall@{k}:    {retrieval_metrics.get('recall_at_k', 0):.2%}")
        print(f"Precision@{k}: {retrieval_metrics.get('precision_at_k', 0):.2%}")
        print(f"F1@{k}:        {retrieval_metrics.get('f1_at_k', 0):.2%}")
        print(f"Hit Rate:    {retrieval_metrics.get('hit_rate', 0):.2%}")
        print(f"MRR:         {retrieval_metrics.get('mrr', 0):.4f}")
        if "ndcg_at_k" in retrieval_metrics:
            print(f"nDCG@{k}:      {retrieval_metrics['ndcg_at_k']:.4f}")
        print(f"{'='*60}\n")

    # Check for revert after attempt 2 (if early switched and quality degraded)
    if state.get("strategy_changed") and state.get("retrieval_attempts") == 1:  # Attempt 2 (0-indexed before increment)
        previous_quality = state.get("previous_quality_score", 0)
        previous_strategy = state.get("previous_strategy")
        current_strategy = state.get("retrieval_strategy")

        # Revert if quality degraded
        if quality_score < previous_quality and previous_strategy:
            print(f"\n{'='*60}")
            print(f"STRATEGY REVERT (NO RE-RETRIEVAL)")
            print(f"Previous ({previous_strategy}): {previous_quality:.0%}")
            print(f"Current ({current_strategy}): {quality_score:.0%}")
            print(f"Strategy degraded quality, reverting (state already has attempt 1 values)")
            print(f"{'='*60}\n")

            # State already has all attempt 1's values!
            # We just need to revert the strategy fields and increment attempts
            return {
                "retrieval_strategy": previous_strategy,  # Revert strategy
                "strategy_switch_reason": f"Reverted to {previous_strategy} (early switch degraded quality)",
                "previous_strategy": None,  # Clear
                "previous_quality_score": None,  # Clear
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,  # Increment
            }

    # Store quality after attempt 1 (before potential early switch)
    previous_quality_updates = {}
    if state.get("retrieval_attempts") == 0:  # Attempt 1 (0-indexed before increment)
        previous_quality_updates = {
            "previous_quality_score": quality_score,
        }

    return {
        "retrieved_docs": [docs_text],
        "retrieval_quality_score": quality_score,
        "retrieval_quality_reasoning": quality_reasoning,
        "retrieval_quality_issues": quality_issues,
        "keywords_to_inject": keywords_to_inject,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "unique_docs_list": unique_docs,
        "retrieval_metrics": retrieval_metrics,
        "messages": [AIMessage(content=f"Retrieved {len(unique_docs)} documents")],
        **previous_quality_updates,
    }

# ============ REWRITING FOR INSUFFICIENT RESULTS ============

def rewrite_and_refine_node(state: dict) -> dict:
    """Inject diagnostic-suggested keywords into query for improved retrieval."""
    query = state["active_query"]
    quality = state.get("retrieval_quality_score", 0)
    keywords = state.get("keywords_to_inject", [])
    issues = state.get("retrieval_quality_issues", [])  # Keep for logging only

    print(f"\n{'='*60}")
    print(f"KEYWORD INJECTION")
    print(f"Original query: {query}")
    print(f"Retrieval quality: {quality:.0%}")
    print(f"Keywords to inject: {keywords}")
    print(f"Issues detected: {', '.join(issues) if issues else 'None'}")
    print(f"{'='*60}\n")

    if not keywords:
        # Fallback: no keywords suggested, return query unchanged
        print(f"No keywords to inject - query unchanged")
        return {"active_query": query}

    # Use revamped rewrite_query from query_optimization.py
    refined_query = rewrite_query(query, keywords)

    print(f"Refined query: {refined_query}")
    print(f"Note: Query expansions cleared - will regenerate for refined query")
    print(f"\nState clearing (keyword injection):")
    print(f"  query_expansions: [] (will regenerate)")
    print(f"  retrieval_query: None (cleared to prevent stale optimization)")
    print(f"  active_query: {refined_query} (with injected keywords)\n")

    return {
        "active_query": refined_query,
        "query_expansions": [],
        "retrieval_query": None,  # Clear stale algorithm optimization
        "messages": [AIMessage(content=f"Query refined with keywords: {query} -> {refined_query}")],
    }

# ============ ANSWER GENERATION & EVALUATION ============

def answer_generation_node(state: dict) -> dict:
    """Generate answer with quality-aware instructions and unified retry handling."""

    question = state["baseline_query"]
    context = state["retrieved_docs"][-1] if state.get("retrieved_docs") else "No context"
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_attempts = state.get("generation_attempts", 0) + 1  # Increment before attempt
    retry_feedback = state.get("retry_feedback", "")

    print(f"\n{'='*60}")
    print(f"ANSWER GENERATION")
    print(f"Question: {question}")
    print(f"Context size: {len(context)} chars")
    print(f"Retrieval quality: {retrieval_quality:.0%}")
    print(f"Generation attempt: {generation_attempts}/3")
    print(f"{'='*60}\n")

    if not context or context == "No context":
        return {
            "final_answer": "I apologize, but I could not retrieve any relevant documents to answer your question. Please try rephrasing your query or check if the information exists in the knowledge base.",
            "generation_attempts": generation_attempts,
            "is_refusal": True,  # Triggers END in route_after_evaluation (skip HHEM/quality eval)
            "is_answer_sufficient": False,
            "messages": [AIMessage(content="Empty retrieval - no answer generated")],
        }

    formatted_context = context

    # Determine quality instruction and retry feedback based on scenario
    # LLM best practices: System prompt = behavioral, User message = content
    retry_feedback_content = ""  # Goes to user message via retry_feedback param

    if generation_attempts > 1 and retry_feedback:
        # RETRY MODE: Split behavioral guidance (system) from content (user message)
        previous_answer = state.get("previous_answer", "")

        # Behavioral guidance only (goes to system prompt)
        quality_instruction = "RETRY: Focus on fixing the issues described in <retry_instructions>. Prioritize factual accuracy over comprehensiveness."

        # Content with previous answer + issues (goes to user message)
        retry_feedback_content = f"""Your previous answer was:
---
{previous_answer}
---

Issues with your previous answer:
{retry_feedback}

Generate an improved answer that fixes ALL issues above. Do not repeat the same unsupported claims."""

        print(f"RETRY MODE:")
        print(f"Feedback:\n{retry_feedback}\n")
    else:
        # First generation - use quality-aware instructions
        if retrieval_quality >= 0.8:
            quality_instruction = f"""High Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents are highly relevant and should contain the information needed to answer the question. Answer directly and confidently based on them."""
        elif retrieval_quality >= 0.6:
            quality_instruction = f"""Medium Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents are somewhat relevant but may have gaps in coverage. Use them to answer what you can, but explicitly acknowledge any limitations or missing information."""
        else:
            quality_instruction = f"""Low Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents may not fully address the question. Only answer what can be directly supported by the context. If the context is insufficient, clearly state: "The provided context does not contain enough information to answer this question completely." """

    spec = get_model_for_task("answer_generation")
    is_gpt5 = spec.name.lower().startswith("gpt-5")

    # System prompt = behavioral guidance, User message = content + retry feedback
    system_prompt, user_message = get_answer_generation_prompts(
        quality_instruction=quality_instruction,
        formatted_context=formatted_context,
        question=question,
        is_gpt5=is_gpt5,
        retry_feedback=retry_feedback_content,  # Content with previous answer + issues
    )

    # Flat low temperature for groundedness preservation
    # Quality improvements come from retry_feedback prompt guidance, not temperature randomness
    # (Variable temp schedule removed after HHEM testing showed 0.7 hurt groundedness)
    temperature = 0.3

    llm = ChatOpenAI(
        model=spec.name,
        temperature=temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])

    result = {
        "final_answer": response.content,
        "generation_attempts": generation_attempts,
        "messages": [response],
    }

    return result

def get_quality_fix_guidance(issues: list[str]) -> str:
    """Generate fix guidance based on quality issues."""
    guidance_map = {
        "incomplete_synthesis": "Provide more comprehensive synthesis of the relevant information",
        "lacks_specificity": "Include specific details (numbers, dates, names, technical terms)",
        "wrong_focus": "Re-read question and address the primary intent",
        "partial_answer": "Ensure all question parts are answered completely",
        "missing_details": "Add more depth and explanation where the context provides supporting information",
    }
    return "; ".join([guidance_map.get(issue, issue) for issue in issues])


def evaluate_answer_node(state: dict) -> dict:
    """Combined refusal + groundedness (HHEM) + quality evaluation with unified retry decision."""
    answer = state.get("final_answer", "")
    # Extract individual chunks for per-chunk HHEM verification (stays under 512 token limit)
    unique_docs = state.get("unique_docs_list", [])
    chunks = [doc.page_content for doc in unique_docs] if unique_docs else []
    question = state["baseline_query"]
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_attempts = state.get("generation_attempts", 0)

    print(f"\n{'='*60}")
    print(f"ANSWER EVALUATION (Refusal + Groundedness + Quality)")
    print(f"Generation attempt: {generation_attempts}")
    print(f"Retrieval quality: {retrieval_quality:.0%}")

    # ==== 1. REFUSAL DETECTION (LLM-as-judge) - Check FIRST ====

    # Detect if LLM refused to answer due to insufficient context
    refusal_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,  # Deterministic for classification
    )
    refusal_checker = refusal_llm.with_structured_output(RefusalCheck)

    try:
        refusal_result = refusal_checker.invoke([{
            "role": "system",
            "content": """You are evaluating whether an AI assistant FULLY REFUSED to answer a question due to insufficient context.

IMPORTANT DISTINCTION:
- FULL REFUSAL: No substantive answer provided, only acknowledges insufficiency (e.g., "The provided context does not contain enough information to answer this question.")
- PARTIAL ANSWER: Provides some information from context BUT acknowledges limitations/gaps (e.g., "Based on the documents, X is true, though the context doesn't provide information about Y.")

The assistant was instructed to use this phrase for FULL REFUSALS:
"The provided context does not contain enough information to answer this question."

ONLY flag as refusal if the answer provides NO substantive information."""
        }, {
            "role": "user",
            "content": f"""Question: {question}

Answer: {answer}

Determine if the assistant FULLY REFUSED (provided NO answer):

FULL REFUSAL indicators (return refused=True):
- Answer provides NO substantive information at all
- Only acknowledges that context is insufficient without providing any facts
- Examples: "I cannot answer this question", "The context doesn't contain this information"

PARTIAL ANSWER indicators (return refused=False):
- Contains ANY facts, details, or explanations from the context
- Even if it acknowledges gaps (e.g., "X uses dropout... however, the learning rate schedule is not covered")
- Provides some useful information even if incomplete

CRITICAL: Presence of limitation phrases does NOT make it a refusal if substantive information is also provided.

Return:
- refused: True ONLY if complete refusal with NO substantive answer
- reasoning: Explain whether answer provided substantive information or only acknowledged insufficiency"""
        }])
        is_refusal = refusal_result["refused"]
        refusal_reasoning = refusal_result["reasoning"]
    except Exception as e:
        print(f"Warning: Refusal detection failed: {e}. Defaulting to not refused.")
        is_refusal = False
        refusal_reasoning = f"Detection failed: {e}"

    print(f"Refusal detection: {'REFUSED' if is_refusal else 'ATTEMPTED'} - {refusal_reasoning}")

    # Early exit if refusal detected (skip expensive checks)
    if is_refusal:
        print(f"Skipping groundedness and quality checks (refusal detected)")
        print(f"{'='*60}\n")
        return {
            "is_refusal": True,  # Terminal state
            "is_answer_sufficient": False,
            "groundedness_score": 1.0,  # Refusal is perfectly grounded (no unsupported claims)
            "has_hallucination": False,
            "unsupported_claims": [],
            "confidence_score": 0.0,
            "answer_quality_reasoning": "Evaluation skipped (LLM refused to answer)",
            "answer_quality_issues": [],
            "retry_feedback": "",
            "messages": [AIMessage(content=f"Refusal detected: {refusal_reasoning}")],
        }

    # ==== 2. GROUNDEDNESS CHECK (HHEM) ====

    # Run HHEM groundedness check with per-chunk verification
    groundedness_result = hhem_detector.verify_groundedness(answer, chunks)
    groundedness_score = groundedness_result.get("groundedness_score", 1.0)
    has_hallucination = groundedness_score < 0.8
    unsupported_claims = groundedness_result.get("unsupported_claims", [])
    print(f"Groundedness: {groundedness_score:.0%} ({hhem_detector.backend_display_name})")

    # EARLY EXIT: Skip quality check if hallucination detected (efficiency optimization)
    if has_hallucination:
        print(f"Hallucination detected - skipping quality check (efficiency optimization)")
        print(f"{'='*60}\n")

        retry_feedback = (
            f"HALLUCINATION DETECTED ({groundedness_score:.0%} grounded):\n"
            f"Unsupported claims: {', '.join(unsupported_claims)}\n\n"
            f"Fix: ONLY state facts explicitly in retrieved context. "
            f"If information is missing, acknowledge the limitation rather than "
            f"adding unsupported details."
        )

        return {
            "is_answer_sufficient": False,
            "groundedness_score": groundedness_score,
            "has_hallucination": True,
            "unsupported_claims": unsupported_claims,
            "retry_feedback": retry_feedback,
            "previous_answer": answer,
            "is_refusal": False,
            # Quality fields: 0.0 (not evaluated, hallucination detected)
            "confidence_score": 0.0,
            "answer_quality_reasoning": "Skipped (hallucination detected)",
            "answer_quality_issues": [],
            "messages": [AIMessage(content=f"Hallucination: {groundedness_score:.0%} grounded, quality check skipped")],
        }

    # ==== 3. QUALITY CHECK (LLM-as-judge) ====
    # Only executes when no hallucination detected

    retrieval_quality_issues = state.get("retrieval_quality_issues", [])
    has_missing_info = any(issue in retrieval_quality_issues for issue in ["partial_coverage", "missing_key_info", "incomplete_context"])
    quality_threshold = 0.5 if (retrieval_quality < 0.6 or has_missing_info) else 0.65

    spec = get_model_for_task("answer_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
    structured_answer_llm = quality_llm.with_structured_output(AnswerQualityEvaluation)

    evaluation_prompt = get_prompt(
        "answer_quality_eval",
        question=question,
        answer=answer,
        retrieval_quality=f"{retrieval_quality:.0%}",
        retrieval_issues=', '.join(retrieval_quality_issues) if retrieval_quality_issues else 'None',
        quality_threshold_pct=quality_threshold*100,
        quality_threshold_low_pct=(quality_threshold-0.15)*100,
        quality_threshold_minus_1_pct=quality_threshold*100-1,
        quality_threshold_low_minus_1_pct=(quality_threshold-0.15)*100-1
    )

    try:
        evaluation = structured_answer_llm.invoke(evaluation_prompt)
        confidence = evaluation["confidence_score"] / 100
        reasoning = evaluation["reasoning"]
        quality_issues = evaluation["issues"]
    except Exception as e:
        print(f"Warning: Answer evaluation failed: {e}. Using conservative fallback.")
        evaluation = {
            "is_relevant": True,
            "is_complete": False,
            "is_accurate": True,
            "confidence_score": 50.0,
            "reasoning": f"Evaluation failed: {e}",
            "issues": ["evaluation_error"]
        }
        confidence = 0.5
        reasoning = evaluation["reasoning"]
        quality_issues = evaluation["issues"]

    is_quality_sufficient = (
        evaluation["is_relevant"] and
        evaluation["is_complete"] and
        evaluation["is_accurate"] and
        confidence >= quality_threshold
    )

    print(f"Quality: {confidence:.0%} ({'sufficient' if is_quality_sufficient else 'insufficient'})")
    if quality_issues:
        print(f"Issues: {', '.join(quality_issues)}")

    # ==== 4. COMBINED DECISION ====
    # Note: If hallucination was detected, we already returned early (line 768)
    # So at this point, has_hallucination is guaranteed to be False

    has_issues = not is_quality_sufficient

    # Build feedback for quality issues (hallucination case handled by early return)
    retry_feedback = ""
    if not is_quality_sufficient:
        # No hallucination, but quality issues: Safe to push for improvements
        retry_feedback = (
            f"QUALITY ISSUES:\n"
            f"Problems: {', '.join(quality_issues)}\n"
            f"Fix: {get_quality_fix_guidance(quality_issues)}"
        )

    print(f"Combined decision: {'RETRY' if has_issues else 'SUFFICIENT'}")
    print(f"{'='*60}\n")

    return {
        "is_answer_sufficient": not has_issues,
        "groundedness_score": groundedness_score,
        "has_hallucination": False,  # Always False here (early exit handles True case)
        "unsupported_claims": [],  # No unsupported claims if no hallucination
        "confidence_score": confidence,
        "answer_quality_reasoning": reasoning,
        "answer_quality_issues": quality_issues,
        "retry_feedback": retry_feedback,
        "previous_answer": answer if has_issues else None,  # Store for retry context
        "is_refusal": False,  # Only reached for non-refusals (early exit handles refusals)
        "messages": [AIMessage(content=f"Evaluation: {groundedness_score:.0%} grounded, {confidence:.0%} quality")],
    }
