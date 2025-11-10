# Introduction: Multihop Reasoning for Question Answering

```{tableofcontents}
```

**Goal**: Implement and evaluate RAG (Retrieval-Augmented Generation) systems for multihop reasoning question answering, focusing on questions that require synthesizing information from multiple documents to arrive at correct answers.

**Dataset**: [HotpotQA Dataset](https://hotpot.ai) - Large-scale dataset specifically designed for multihop reasoning over text

**Focus**: Multihop reasoning challenges where answers require connecting information across multiple supporting documents

:::{iframe} https://www.youtube.com/embed/T-D1OfcDW1M
:width: 100%
:align: center

What is Retrieval-Augmented Generation (RAG)? - IBM Technology explains how RAG combines retrieval and generation for improved AI responses
:::

---

## The Multihop Reasoning Challenge

```{admonition} What is Multihop Reasoning?
:class: note
**Multihop reasoning** in question answering refers to the ability to synthesize information from multiple documents or passages to answer complex questions. Unlike single-hop QA where answers can be found in one document, multihop questions require connecting facts from different sources to arrive at the correct answer.
```

Multihop reasoning represents one of the most challenging frontiers in question answering systems, where we tackle complex questions that cannot be answered by simply extracting information from a single document. Instead, these questions require sophisticated reasoning that connects information across multiple sources, performs logical inference, and synthesizes evidence to construct comprehensive answers.

The core challenge involves developing systems that can identify relevant documents, extract supporting evidence from multiple sources, and combine this information through logical reasoning to generate accurate answers. Unlike single-hop QA systems, multihop reasoning requires sophisticated orchestration of retrieval, evidence selection, and reasoning components working together seamlessly.

---

## Why This Matters

````{grid} 2
```{grid-item-card} Technical Innovation
:class-header: bg-primary text-white

- Advances complex reasoning capabilities in AI systems
- Develops evaluation metrics for multi-document understanding
- Pushes boundaries of information synthesis from multiple sources
```

```{grid-item-card} Real-World Impact
:class-header: bg-success text-white

- Accelerates research and knowledge discovery
- Improves fact-checking and verification systems
- Enhances educational and decision support tools
- Enables more sophisticated information access
```
````

From a technical perspective, multihop reasoning for question answering represents one of the most challenging frontiers in natural language understanding. Unlike simpler QA tasks where answers can be extracted from single passages, multihop questions require systems to identify relevant information scattered across multiple documents, establish connections between disparate facts, and synthesize these connections to construct accurate answers. This challenge pushes the boundaries of current retrieval and reasoning architectures, requiring sophisticated approaches that can handle complex information flow across document boundaries.

```{note}
Mastering multihop reasoning capabilities opens doors to practical applications across diverse domains—from research assistance and fact-checking to educational systems and decision support tools. Systems that can reliably answer complex questions requiring synthesis of multiple sources represent a significant step toward truly intelligent information retrieval.
```

---

## Real-World Applications

```{dropdown} Research Assistance & Literature Review
Enable researchers across domains to quickly identify connections between disparate studies, discover related work that spans multiple sub-fields, and synthesize knowledge from comprehensive literature searches. Multihop QA systems can answer complex questions like "What methodology did the team that discovered X use in their follow-up study?" requiring connections across multiple papers.
```

```{dropdown} Fact-Checking & Verification
Support journalists, fact-checkers, and content moderators by answering complex verification questions that require cross-referencing multiple sources. For example, "Did the politician who voted against Bill A also serve on the committee that drafted it?" requires reasoning across voting records and committee membership data.
```

```{dropdown} Educational & Learning Systems
Provide students and educators with intelligent tutoring systems capable of answering sophisticated questions that require synthesizing information from multiple textbook chapters, lecture materials, or knowledge sources. Questions like "How does the concept introduced in Chapter 3 relate to the theorem proved in Chapter 7?" demand multihop reasoning.
```

```{dropdown} Business Intelligence & Decision Support
Assist analysts and executives in answering strategic questions that span multiple data sources and reports. Questions such as "Which product line managed by the VP who joined last year had the highest growth?" require connecting organizational, product, and performance data across multiple documents.
```

```{dropdown} Legal Research & Case Analysis
Accelerate legal research by enabling queries that connect precedents, statutes, and case law across multiple jurisdictions and time periods. Attorneys can ask "What was the precedent cited by the judge who ruled on the similar case in the neighboring district?" requiring multidocument synthesis.
```

---

## The HotpotQA Dataset: A Multihop Reasoning Case Study

````{grid} 1 2 2 3
:gutter: 3

```{grid-item-card} Scale & Coverage
:class-header: bg-primary text-white

- **~90,000** training examples
- **~7,400** validation examples
- **Wikipedia** as knowledge source
- **Crowdsourced** annotations
```

```{grid-item-card} Question Types
:class-header: bg-secondary text-white

- Bridge Entity Questions (~80%)
- Comparison Questions (~20%)
- Multi-document Synthesis
- Complex Reasoning Chains
```

```{grid-item-card} Reasoning Patterns
:class-header: bg-success text-white

- Entity relationship queries
- Temporal comparison questions
- Attribute-based comparisons
- Multi-step inference requirements
```

```{grid-item-card} Dataset Characteristics
:class-header: bg-info text-white

- Genuine multihop requirements
- Supporting facts annotations
- Distractor paragraphs included
- Diverse Wikipedia topics
```
````

```{admonition} HotpotQA as Multihop Reasoning Benchmark
:class: important

The HotpotQA dataset serves as the premier benchmark for multihop question answering, specifically designed to test systems' ability to reason across multiple documents. Unlike simpler QA datasets where answers can be found in single passages, HotpotQA requires synthesizing information from multiple Wikipedia articles, with explicit annotations of supporting facts that enable both answer generation and reasoning path evaluation.
```

### Dataset Characteristics Impacting Method Selection

Our analysis reveals key dataset properties that influence the effectiveness of different retrieval approaches for multihop reasoning:

- **Multihop Requirement**: 95%+ questions require evidence from 2 or more distinct Wikipedia articles
- **Bridge Entity Complexity**: Most questions involve intermediate entities that connect the question to the final answer
- **Distractor Challenge**: 80% of provided context consists of plausible but irrelevant passages, testing retrieval robustness
- **Evidence Requirements**: Answers must be traceable to specific supporting facts, enabling evaluation of reasoning transparency

---

## Our RAG Pipeline Architecture

This project implements a complete RAG pipeline for HotpotQA multihop question answering, focusing on three core components:

````{grid} 3
```{grid-item-card} 1. Retrieval System
:class-header: bg-primary text-white

**Vector Search + Reranking**
^^^
- Dense retrieval with BGE embeddings
- Cross-encoder reranking for precision
- Top-k passage selection (k=8-12)
- Handles multi-document evidence gathering
```

```{grid-item-card} 2. Generator Fine-tuning
:class-header: bg-success text-white

**QLoRA for Mistral-7B**
^^^
- 4-bit quantization (NF4)
- Low-rank adaptation (r=64)
- Parameter-efficient fine-tuning
- Trains answer generation + citation
```

```{grid-item-card} 3. Training Strategy
:class-header: bg-info text-white

**Curriculum Learning**
^^^
- Synthetic reasoning generation
- Progressive difficulty scaling
- Explicit edge case handling
- Insufficient context detection
```
````

### Pipeline Stages

````{tab-set}
```{tab-item} Stage 1: Retrieval
**Goal**: Gather relevant passages from multiple documents

**Components**:
- **Dense Retriever**: BGE-large embeddings for semantic search across Wikipedia passages
- **Reranker**: Cross-encoder (BGE reranker) to score and refine top candidates
- **Output**: Top 8-12 most relevant passages per question

**Why This Matters**: Multihop questions require evidence from multiple documents. Dense retrieval captures semantic similarity, while reranking ensures precision by comparing question-passage pairs directly.
```

```{tab-item} Stage 2: Generation
**Goal**: Generate concise answers with accurate citations

**Components**:
- **Base Model**: Mistral-7B-Instruct (7B parameters)
- **Quantization**: 4-bit NF4 quantization for memory efficiency
- **Fine-tuning**: QLoRA adapters (rank=64) on HotpotQA training data
- **Output Format**: Structured JSON with reasoning, answer, and citations

**Why This Matters**: Base LLMs struggle with citation accuracy and edge case handling. Fine-tuning teaches the model to generate concise extractive answers and identify when context is insufficient.
```

```{tab-item} Stage 3: Training Strategy
**Goal**: Maximize answer quality and reliability

**Key Techniques**:
1. **Synthetic Reasoning Generation**: Create training examples from gold supporting facts
2. **Curriculum Learning**: Progressively increase retrieval difficulty (epochs 1-2: gold + distractors → epochs 3+: realistic retrieval)
3. **Edge Case Training**: 15-20% "insufficient context" examples to prevent hallucination
4. **Concise Chain-of-Thought**: Keep reasoning <100 tokens for training stability

**Why This Matters**: Training data quality determines success. These strategies enable the model to learn robust citation patterns and admit uncertainty when appropriate.
```
````

### Key Innovation: Fine-tuning Strategy

Our approach focuses on **teaching the generator to handle multihop reasoning reliably** through careful training data design:

```{admonition} Training Data Construction
:class: important

**Baseline Approach** (Prompting Only):
- Zero-shot prompting with retrieval results
- **Result**: 17.5% EM, 27.4% F1, 11.7% insufficient context detection

**Our Approach** (QLoRA Fine-tuning):
- 500 training examples with synthetic reasoning chains
- Explicit citation training [1], [2] embedded in reasoning
- Progressive curriculum from easy → realistic retrieval
- Edge case examples for "insufficient context"
- **Result**: 41.5% EM, 46.4% F1, 60.0% insufficient context detection

**Key Insight**: Fine-tuning enables behaviors nearly impossible through prompting alone, especially admitting uncertainty when context is insufficient (+412.8% improvement).
```

### Evaluation Framework for Multihop QA

We establish comprehensive evaluation criteria specifically designed for multihop question answering systems:

- **Answer Correctness**: F1 and Exact Match metrics measuring answer quality
- **Supporting Fact Identification**: Precision, recall, and F1 for evidence selection
- **Reasoning Transparency**: Ability to trace reasoning chains across documents
- **Citation Accuracy**: Correct identification of supporting passages
- **Robustness**: Performance across different question types (bridge, comparison) and difficulty levels

---

## What You'll Learn

This tutorial demonstrates end-to-end RAG system development for multihop question answering:

1. **Complete RAG Pipeline**: Dense retrieval (BGE embeddings) → Cross-encoder reranking → QLoRA-tuned generation
2. **Fine-tuning Strategy**: Curriculum learning, synthetic reasoning generation, and edge case handling for reliable multihop QA
3. **Comprehensive Evaluation**: 6-metric framework (Answer F1/EM, Citation Precision/Recall/F1, Insufficient Context Detection)
4. **Practical Implementation**: Colab-friendly QLoRA training with 4-bit quantization for Mistral-7B on T4/A100 GPUs
5. **Real Results**: Baseline (17.5% EM) → Fine-tuned (41.5% EM) with +412.8% improvement in edge case handling

---

```{epigraph}
"The ability to reason across multiple documents represents a fundamental leap toward human-like information synthesis. Systems that can connect disparate facts, trace reasoning chains, and admit uncertainty are not merely answering questions—they are demonstrating genuine understanding of how knowledge interconnects."

-- *The Evolution of AI Question Answering Systems*
```

```{seealso}
Ready to build your own multihop RAG system? Follow the tutorial sequence:
- **Exploratory Data Analysis** - Understand HotpotQA structure, question types, and multihop patterns
- **Evaluation Framework** - Implement 6 generation quality metrics for comprehensive RAG assessment
- **Retrieval System** - Dense retrieval with BGE embeddings, cross-encoder reranking, and baseline evaluation
- **Generator Fine-tuning** - QLoRA fine-tuning pipeline with curriculum learning and synthetic data generation
- **Results Comparison** - Detailed analysis of baseline vs. fine-tuned performance across all metrics
- **Conclusion** - Key learnings, failure analysis, and practical insights from development
```
