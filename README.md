# Advanced Agentic RAG using LangGraph

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain 1.0](https://img.shields.io/badge/LangChain-1.0-green.svg)
![LangGraph 1.0](https://img.shields.io/badge/LangGraph-1.0-purple.svg)

An advanced Agentic RAG system that autonomously adapts its retrieval strategy and reasoning process through dynamic decision-making, iterative self-correction, and intelligent tool selection. Built with LangGraph's StateGraph pattern, the system embeds autonomous reasoning into a 7-node workflow where routing functions and conditional edges provide distributed intelligence--no central "agent" orchestrator needed.

## Demo

https://github.com/user-attachments/assets/c4168ac9-3eb0-45dc-be67-895299d8a97e

> The walkthrough above tours the agent's graph in LangGraph Studio--its 7 nodes, conditional
> routing, and self-correction loops. There is no hosted live endpoint: the Azure stack was
> provisioned, verified end-to-end against the golden dataset, and then torn down to keep idle
> cost at ~$0 (a portfolio project with no sustained traffic). You can run the entire system
> locally in minutes (see [Quick Start](#quick-start)), or redeploy the full Azure stack (see
> [Deployment & Cost](#deployment--cost)).

## Key Results

- **83% retrieval improvement** (F1@4: 17.3% → 31.7%) with budget models only
- Demonstrates architectural value independent of model quality
- [Full evaluation details](#evaluation)

## Table of Contents

- [Why This Qualifies as Agentic RAG](#why-this-qualifies-as-agentic-rag)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Architecture Tiers](#architecture-tiers)
- [Quick Start](#quick-start)
- [Model Tier Configuration](#model-tier-configuration)
- [Technology Stack](#technology-stack)
- [Deployment & Cost](#deployment--cost)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)

## Why This Qualifies as Agentic RAG

This system demonstrates all four core characteristics that define Agentic RAG:

### 1. Autonomous Decision-Making
**Traditional RAG**: Fixed pipeline (query -> retrieve -> generate)<br>
**This System**: Dynamic routing based on quality evaluation at 2 decision points

The system autonomously plans next steps based on intermediate results:
- `route_after_retrieval`: Decides to proceed, rewrite query, or switch strategy based on quality scores
- `route_after_evaluation`: Evaluates answer quality, decides to retry generation or return result

### 2. Iterative Self-Correction
**Traditional RAG**: Single-pass retrieval and generation<br>
**This System**: Two self-correction loops with quality gates

- **Retrieval Loop**: Poor quality (<0.6) -> 5 issue types + keyword injection -> rewrite query or switch strategy (max 2 attempts)
- **Generation Loop**: Consolidated evaluation (refusal + HHEM hallucination + quality) -> unified feedback -> regenerate with low temperature (max 3 attempts)
- **Early Strategy Switching**: off_topic/wrong_domain detected -> immediate strategy switch (saves 30-50% tokens)

### 3. Context Management
**Traditional RAG**: Stateless, no conversation memory<br>
**This System**: Persistent state across conversation turns

- Conversational rewrite transforms follow-up queries into self-contained questions
- MemorySaver checkpointer persists state across multi-turn conversations

### 4. Intelligent Tool Selection
**Traditional RAG**: Single retrieval method<br>
**This System**: Three retrieval strategies with intelligent selection

- **Strategies**: Semantic (FAISS), Keyword (BM25), Hybrid (RRF fusion)
- **Selection**: `decide_retrieval_strategy_node` analyzes corpus stats + query characteristics
- **Adaptation**: Switches strategies mid-execution based on content analysis

### Architecture Pattern

**No Central Agent Orchestrator**: The LangGraph StateGraph itself IS the agent. Decision-making is distributed across routing functions and conditional edges. This "Dynamic Planning and Execution Agents" pattern is more controllable and debuggable than single-agent orchestration while maintaining full autonomy through quality-driven routing.

### Research-Backed Enhancements

- **CRAG**: Confidence-based action triggering with early detection at retrieval stage
- **PreQRAG**: Strategy-specific query optimization (13-14% MRR improvement)
- **RAG-Fusion**: Multi-query retrieval with RRF ranking fusion (3-5% MRR improvement)
- **vRAG-Eval**: Answer quality evaluation with adaptive thresholds (65%/50% based on retrieval quality)
- **Hallucination Detection**: Claim decomposition + HHEM verification with pluggable backends (local HHEM-2.1-Open or Vectara HHEM-2.3 API)

## Architecture Overview

The system uses a 7-node LangGraph workflow with autonomous decision-making at every stage. Quality gates and routing functions provide distributed intelligence--each decision point evaluates intermediate results and autonomously determines the next action.

### Advanced RAG (7 nodes, 2 routers, 2 self-correction loops)

![Advanced RAG Architecture](mermaid%20chart.png)

**See [Interactive Demo](Advanced_Agentic_RAG.ipynb)** for routing logic deep-dive and live comparison runs.

### Node Summary

| Node | Purpose |
|------|---------|
| `conversational_rewrite` | Makes query self-contained using conversation history |
| `decide_strategy` | Selects optimal retrieval strategy (semantic/keyword/hybrid) |
| `query_expansion` | Generates query variations, optimizes for strategy |
| `retrieve_with_expansion` | RRF fusion + two-stage reranking + quality evaluation |
| `rewrite_and_refine` | Query enrichment via keyword injection for improved retrieval |
| `answer_generation` | Structured RAG prompting with quality-aware instructions |
| `evaluate_answer` | Consolidated refusal + HHEM hallucination + quality assessment |

## Features

The Advanced tier implements 17 features across retrieval, generation, and evaluation.

<details>
<summary><strong>Document & Corpus Profiling</strong></summary>

- LLM-based profiling of documents
- Analyzes technical density, document types, and domain characteristics
- Informs retrieval strategy selection
</details>

<details>
<summary><strong>Query Processing</strong></summary>

- **Conversational Rewriting**: Transforms follow-up queries into self-contained questions
- **Query Expansion**: Generates 3 variations with query-type detection (comparison, adaptation, other)
- **Strategy-Specific Optimization**: Keyword (exact terms) / Semantic (conceptual) / Hybrid (balanced)
</details>

<details>
<summary><strong>Intelligent Strategy Selection</strong></summary>

- Pure LLM-based classification (domain-agnostic, handles all edge cases)
- Analyzes query characteristics + corpus statistics
- Selects semantic/keyword/hybrid with confidence score + reasoning
</details>

<details>
<summary><strong>Multi-Strategy Retrieval</strong></summary>

- **Semantic**: FAISS vector search for meaning-based retrieval
- **Keyword**: BM25 lexical search for exact term matching
- **Hybrid**: Combines both with RRF-based fusion
- **RRF Multi-Query Fusion**: Aggregates rankings across query variants BEFORE reranking
</details>

<details>
<summary><strong>Two-Stage Reranking</strong></summary>

- **Stage 1**: CrossEncoder (ms-marco-MiniLM-L-6-v2) filters to top-10
- **Stage 2**: LLM-as-judge scores each document 0-100, selects top-4
- 3-5x faster than pure LLM reranking, 5-10x cheaper
</details>

<details>
<summary><strong>Quality Gates & Self-Correction</strong></summary>

- **Retrieval Quality**: 5 issue types (partial_coverage, missing_key_info, incomplete_context, wrong_domain, off_topic) + keyword injection
- **Answer Quality**: 5 issue types (incomplete_synthesis, lacks_specificity, missing_details, partial_answer, wrong_focus)
- **Adaptive Thresholds**: 65% for good retrieval, 50% for poor retrieval
</details>

<details>
<summary><strong>HHEM-Based Hallucination Detection</strong></summary>

- Pluggable backend architecture: Local HuggingFace model (HHEM-2.1-Open) or Vectara managed API (HHEM-2.3)
- Claim decomposition: LLM extracts individual claims from answers
- HHEM verification: Each claim validated against retrieved contexts
- Groundedness threshold: 0.8 (scores below 80% trigger regeneration)
- Backend selection via HHEM_BACKEND environment variable (local/vectara)
</details>

<details>
<summary><strong>Multi-turn Conversations</strong></summary>

- Preserves conversation context with state persistence
- Thread management via MemorySaver checkpointer
- Automatic query contextualization
</details>

## Architecture Tiers

All tiers use the same **budget model tier** (GPT-4o-mini) to isolate architectural improvements from model quality.

| Tier | Features | Key Additions |
|------|----------|---------------|
| **Basic** | 1 | Semantic search only, direct LLM generation |
| **Intermediate** | 5 | + Query expansion, hybrid retrieval, CrossEncoder reranking, RRF fusion |
| **Advanced** | 17 | + Strategy selection, two-stage reranking, HHEM detection, quality gates, self-correction loops |
| **Multi-Agent** | 20 | + Query decomposition, parallel retrieval workers, cross-agent LLM relevance scoring |

**Run the comparison yourself:** See [Advanced_Agentic_RAG.ipynb](Advanced_Agentic_RAG.ipynb)

### When to Use Each Tier

- **Basic**: Simple factual lookups, low latency requirements
- **Intermediate**: Enhanced retrieval for predictable latency
- **Advanced**: Complex domains where query understanding matters
- **Multi-Agent**: Research synthesis, multi-faceted questions

## Quick Start

**Prerequisites:** Python 3.11+

```bash
# 1. Install dependencies (uses uv, not pip)
uv sync

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Run demo
uv run python main.py
```

## Model Tier Configuration

Control cost-quality tradeoffs via `MODEL_TIER` environment variable:

| Tier | Models | Use Case |
|------|--------|----------|
| **budget** | All GPT-4o-mini | Development, demos, architecture showcase |
| **balanced** | GPT-4o-mini + GPT-5-mini | Production (cost-conscious) |
| **premium** | GPT-5.1 + GPT-5-mini + GPT-5-nano | Production (quality-critical) |
 
```bash
# Set in .env
MODEL_TIER=budget    # Default - best cost-efficiency
MODEL_TIER=balanced  # Best cost-quality tradeoff
MODEL_TIER=premium   # Maximum quality
```

## Technology Stack

- **LLM Framework**: LangChain 1.0
- **Orchestration**: LangGraph 1.0 (StateGraph)
- **Vector Store**: FAISS
- **Lexical Search**: BM25
- **LLMs**: OpenAI GPT-4o-mini/GPT-5-mini/GPT-5.1/GPT-5-nano (configurable)
- **PDF Processing**: Marker (layout-aware, table/figure extraction, OCR-capable)
- **Reranking**: sentence-transformers (CrossEncoder)
- **Hallucination Detection**: HHEM-2.1-Open (local) or HHEM-2.3 (Vectara managed API)
- **Package Manager**: uv

## Deployment & Cost

The system was deployed to **Azure Container Apps**, with Container Registry, Key Vault, and
Application Insights--all defined as infrastructure-as-code in [`infra/main.bicep`](infra/main.bicep).
(The live stack has since been torn down--see the [Demo](#demo) note.)

Redeploy the full Azure stack from the Bicep in `infra/` with a single command:

```bash
./scripts/deploy-infra.sh   # provision all Azure resources via Bicep
# then run the "Deploy to Azure Container Apps" workflow (manual trigger) to build & deploy
```

The deployed configuration uses `HHEM_BACKEND=vectara` (Vectara's managed HHEM-2.3 API) for
hallucination detection, so a redeploy needs `VECTARA_API_KEY` / `VECTARA_CUSTOMER_ID`. For
fully-offline runs, `HHEM_BACKEND=local` swaps in HHEM-2.1-Open (baked into the image),
requiring only your `OPENAI_API_KEY`.

## Evaluation

### Metrics

- **Retrieval**: F1@K, Precision@K, Recall@K, MRR, nDCG
- **Generation**: Groundedness (HHEM-based), Semantic Similarity, Factual Accuracy, Completeness

### Golden Datasets

| Dataset | Questions | Avg Chunks | Cross-Doc | Query Types |
|---------|-----------|------------|-----------|-------------|
| **Standard** | 20 | 2.1 | 10% | factual, conceptual, procedural, comparative |
| **Hard** | 10 | 4.7 | 50% | procedural, comparative (multi-document) |

### Architecture Comparison Results

All tiers use **budget models** (GPT-4o-mini only) to isolate architectural improvements from model quality.

#### Standard Dataset (20 questions, k=4)

| Tier | F1@4 | MRR | nDCG@4 | Groundedness |
|------|------|-----|--------|--------------|
| Basic | 17.3% | 0.254 | 0.236 | 48.6% |
| Intermediate | 22.7% | 0.450 | 0.343 | 70.7% |
| Advanced | 29.3% | 0.600 | 0.484 | 64.1% |
| **Multi-Agent** | **31.7%** | **0.600** | **0.497** | **76.6%** |

*Maximum achievable F1@4 is 64.6% (dataset avg: 2.1 relevant docs/question). Multi-Agent achieves 49% of ceiling.*

#### Hard Dataset (10 questions, k=6, multi-document)

| Tier | F1@6 | MRR | nDCG@6 | Groundedness |
|------|------|-----|--------|--------------|
| Basic | 22.0% | 0.458 | 0.300 | 60.4% |
| Intermediate | 25.6% | 0.408 | 0.293 | 62.5% |
| Advanced | 32.5% | **0.750** | 0.460 | **88.9%** |
| **Multi-Agent** | **38.7%** | 0.633 | **0.480** | 87.0% |

*Maximum achievable F1@6 is 84.8% (dataset avg: 4.7 relevant docs/question). Multi-Agent achieves 46% of ceiling.*

## Future Improvements

- **HyDE**: Hypothetical document embeddings for better retrieval
- **Step-back prompting**: Higher-level conceptual questions for multi-hop reasoning
- **Chain-of-thought generation**: Structured reasoning with mandatory inline citations
- **Context compression**: Reduce prompt tokens by 75% while maintaining accuracy
- **LangSmith integration**: Production tracing, user feedback collection, quality dashboards
