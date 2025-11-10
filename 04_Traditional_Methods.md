# Chapter 4 - Traditional Information Retrieval Methods

```{tableofcontents}
```

:::{iframe} https://www.youtube.com/embed/a3sg6MH8m4k
:width: 100%
:align: center

BM25 Algorithm Explained - Understanding the Best Matching 25 ranking function for information retrieval
:::

## Table of Contents

### 4.1 Traditional IR Algorithms Selection
- BM25 and DPR method selection rationale
- Why this approach qualifies as "Traditional"
- Core approach and mathematical foundations

### 4.2 Method Analysis and Trade-offs
- Advantages for multihop reasoning applications
- Limitations motivating modern methods
- Comparison with alternative approaches

### 4.3 Implementation Framework
- Libraries and frameworks (HuggingFace + rank-bm25)
- Off-the-shelf components and HotpotQA preprocessing
- Lightweight reader architecture components

### 4.4 Implementation Strategy
- BM25/DPR retrieval pipeline workflow
- BGE reranking integration
- Mistral-7B answer generation pipeline

---

## 4.1 Choose Traditional IR Algorithms as Baseline Methods

We selected **BM25 (Best Matching 25)** and **DPR (Dense Passage Retrieval)** as our retrieval methods, combined with a lightweight reader architecture using **BAAI/bge-reranker-base** for reranking and **Mistral-7B-Instruct** for answer generation. This approach qualifies as "traditional" because it involves **no fine-tuning** of any components.

:::{admonition} Why This Approach Qualifies as "Traditional"
:class: note
This approach is considered "traditional" because it uses **off-the-shelf, pre-trained models without any fine-tuning**:
- **No domain-specific training** (models used as-is from HuggingFace)
- **Zero-shot application** (no parameter updates on HotpotQA data)
- **Standard retrieval pipeline** (retrieve → rerank → generate)
- **Baseline methodology** (establishes performance floor before deep learning enhancements)
- **No end-to-end optimization** (components work independently without joint training)
:::

### Method Overview

**Core Approach:**
- **BM25 Retrieval**: Sparse retrieval using term frequency statistics for initial candidate selection
- **DPR Vector Search**: Dense passage retrieval using vector similarity (cosine similarity/inner product) for semantic matching
- **BGE Reranking**: Cross-encoder reranker (BAAI/bge-reranker-base) refines top candidates
- **Mistral Generation**: Mistral-7B-Instruct generates final answers from reranked passages

:::{admonition} Vector Search in Our Pipeline
:class: important
**DPR is Vector Search!** Dense Passage Retrieval (DPR) implements vector search through:
- **Query Encoding**: Question → Dense vector embedding
- **Document Encoding**: Passages → Dense vector embeddings  
- **Similarity Search**: Maximum Inner Product Search (MIPS) / Cosine similarity
- **Top-k Retrieval**: Return most similar vectors (passages) to query vector

This is the same underlying technology as modern vector databases (Pinecone, Chroma, Weaviate).
:::

**Pipeline Architecture:**
```
Query → BM25/DPR Retrieval → BGE Reranking → Mistral Generation → Answer
```

### Pipeline Logic for HotpotQA Multihop Reasoning

The pipeline design specifically addresses the core challenges of multihop reasoning in HotpotQA:

:::{admonition} Multihop QA Pipeline Requirements
:class: important

**HotpotQA Challenge**: Questions require synthesizing information from multiple Wikipedia passages to arrive at correct answers, often involving bridge entities and comparative reasoning.

**Pipeline Solution**: A 4-stage approach that progressively narrows down information while maintaining reasoning capability:

1. **Efficient Retrieval** (BM25 + DPR): Cast a wide net to find potentially relevant passages from millions of Wikipedia articles
2. **Intelligent Selection** (BGE Reranker): Identify the most promising passages for reasoning
3. **Information Synthesis** (Mistral-7B): Perform actual reasoning and generate answers
4. **Evidence Tracking**: Maintain traceability for supporting fact identification
:::

**Why Each Component is Essential:**

**Sparse Retrieval (BM25) - Entity Matching Foundation**
- **Purpose**: Find passages containing entities mentioned in the question
- **HotpotQA Strength**: Excels at bridge entity questions where specific entities must be found
- **Example**: For "Which magazine was started first, Arthur's Magazine or First for Women?", BM25 ensures passages about both magazines are retrieved
- **Efficiency**: Handles millions of Wikipedia passages with millisecond response time

:::{iframe} https://www.youtube.com/embed/iMqqY-_j0ao
:width: 100%
:align: center

Dense Passage Retrieval (DPR) - Learning semantic representations for neural information retrieval
:::

**Dense Retrieval (DPR) - Semantic Understanding**
- **Purpose**: Capture semantic relationships beyond exact keyword matching
- **HotpotQA Strength**: Finds passages related to question intent even without exact entity overlap
- **Example**: Can find passages about publication dates and founding information even if phrased differently
- **Complementarity**: Works with BM25 to create comprehensive candidate set

**Reader/Reranker (BGE) - Intelligent Passage Selection via Cross-Encoder**
- **Purpose**: Identify which passages actually contain information needed for multihop reasoning
- **Cross-Encoder Architecture**: Uses BAAI/bge-reranker-base cross-encoder for joint query-passage attention
- **HotpotQA Strength**: Understands query-passage relevance at a deeper level than simple similarity
- **Example**: Distinguishes between passages that contain the entities vs. passages that contain answerable information about those entities
- **Efficiency**: Reduces 100+ candidates to 10 most promising passages for reasoning

**Cross-Encoder vs Bi-Encoder for Reranking:**

:::{iframe} https://www.youtube.com/embed/OATCgQtNX2o
:width: 100%
:align: center

Cross-Encoders vs Bi-Encoders - Understanding the trade-offs between accuracy and efficiency in neural ranking
:::

:::{admonition} Cross-Encoder Advantage in Reranking
:class: tip

**Cross-Encoder (BGE Reranker)**:
- **Joint Processing**: Query and passage processed together through shared Transformer layers
- **Full Attention**: Query tokens can attend to all passage tokens and vice versa
- **Superior Accuracy**: More precise relevance scoring due to complete interaction modeling
- **Computational Cost**: Slower inference (each query-passage pair requires separate forward pass)

**Bi-Encoder (DPR Style)**:
- **Separate Processing**: Query and passage encoded independently
- **Limited Interaction**: No cross-attention between query and passage
- **Efficiency**: Fast inference via pre-computed embeddings and dot-product similarity
- **Accuracy Trade-off**: Less precise due to lack of query-passage interaction

**Why Cross-Encoder for Reranking**: After bi-encoder retrieval casts a wide net, cross-encoder reranking provides the accuracy boost needed for final passage selection in multihop reasoning tasks.
:::

**Language Model (Mistral-7B) - Reasoning and Synthesis**
- **Purpose**: Perform the actual multihop reasoning and answer synthesis
- **HotpotQA Strength**: Can connect information across multiple passages to derive answers
- **No Fine-tuning**: Uses pre-trained instruction-following capabilities to handle complex reasoning
- **Example**: Reads passages about both magazines, compares founding dates, and determines which was "first"

### Concrete Pipeline Example: Bridge Entity Question

**Question**: "What position did the player drafted by the Boston Celtics in 1981 play?"

**Stage 1 - Hybrid Retrieval (BM25 + DPR)**:
- **BM25 finds**: Passages containing "Boston Celtics", "1981", "drafted"
- **DPR finds**: Passages about NBA drafts, player positions, basketball terminology
- **Result**: 100+ candidate passages from Wikipedia, including ones about the 1981 NBA Draft

**Stage 2 - BGE Cross-Encoder Reranking**:
- **Cross-Attention Processing**: Each (query, passage) pair processed jointly through Transformer layers
- **Deep Relevance Understanding**: Cross-encoder models query-passage interactions for multihop reasoning potential
- **Identifies**: Most relevant passages likely to contain both draft information AND player position
- **Result**: Top 10 passages, prioritizing those with both draft details and player information based on cross-encoder scores

**Stage 3 - Mistral-7B Reasoning**:
- **Reads**: "Charles Bradley was drafted by Boston Celtics in 1981..." and "Charles Bradley played center..."
- **Connects**: Draft information with position information across passages
- **Synthesizes**: "center" as the final answer

**Stage 4 - Evidence Tracking**:
- **Maintains**: References to supporting passages for evaluation
- **Enables**: Assessment against HotpotQA's Joint EM metric (answer + supporting facts)

:::{note}
**Why This Works Without Fine-tuning**: Each component leverages pre-trained capabilities (BM25 statistics, DPR embeddings, BGE cross-attention, Mistral instruction-following) while the pipeline design handles the multihop reasoning challenge through intelligent information flow rather than learned end-to-end optimization.
:::

**Mathematical Foundations:**

**BM25 Scoring:**

```{math}
\text{BM25}(q,d) = \sum_{i=1}^{n} \text{IDF}(q_i) \times \frac{f(q_i,d) \times (k_1 + 1)}{f(q_i,d) + k_1 \times \left(1 - b + b \times \frac{|d|}{\text{avgdl}}\right)}
```

**DPR Similarity:**

```{math}
\text{sim}(q,p) = \text{E}_q^T \cdot \text{E}_p
```

Where $E_q$ and $E_p$ are pre-trained dense embeddings for query and passage.

**BGE Cross-Encoder Reranking Score:**

```{math}
\text{score}(q,p) = \text{softmax}(\text{W} \cdot \text{BGE}_{\text{cross-encoder}}([\text{CLS}; q; \text{SEP}; p; \text{SEP}]))
```

Where:
- $[\text{CLS}; q; \text{SEP}; p; \text{SEP}]$ represents the concatenated input sequence
- $\text{BGE}_{\text{cross-encoder}}$ applies full cross-attention between query and passage tokens
- $\text{W}$ is a learned classification head that outputs relevance score
- Cross-attention allows query tokens to directly interact with passage tokens

**Cross-Encoder vs Bi-Encoder Mathematical Comparison:**

**Bi-Encoder (DPR-style)**:
```{math}
\text{score}(q,p) = \text{E}_q^T \cdot \text{E}_p
```

**Cross-Encoder (BGE-style)**:
```{math}
\text{score}(q,p) = f_{\theta}(\text{MultiHeadAttention}(q \oplus p))
```

Where $q \oplus p$ represents joint processing with full cross-attention capabilities.

## 4.2 Explain the Methods and Pros/Cons

### ✅ Advantages (Why We Chose These Methods)

:::{tab-set}
```{tab-item} Multihop QA Strengths
- **Entity matching** - BM25 excels at finding entities mentioned in questions
- **Hybrid retrieval** - Combines sparse (BM25) and dense (DPR) strengths
- **Interpretable results** - can trace why specific passages are retrieved
- **Fast processing** - suitable for large Wikipedia collections
- **No training data** - works immediately with HotpotQA without fine-tuning
```

```{tab-item} Reliability & Efficiency
- **Proven effectiveness** - BM25 is standard for entity-based retrieval
- **Computational efficiency** - faster than fully neural end-to-end methods
- **Scalable architecture** - handles millions of Wikipedia articles
- **Robust performance** - consistent results across different question types
- **Lightweight reader** - BGE reranker + Mistral-7B provides good answer quality
```

```{tab-item} Research Value
- **Strong baselines** - establishes performance floor for modern methods
- **Transparent operation** - easy to analyze failure cases in multihop reasoning
- **Component modularity** - can replace individual components for ablation studies
- **Comparative analysis** - reveals advantages of end-to-end fine-tuned approaches
```
:::

### ❌ Disadvantages (Motivating Modern Methods)

:::{tab-set}
```{tab-item} Reasoning Limitations
- **No end-to-end optimization** - components not jointly trained for multihop reasoning
- **Limited reasoning understanding** - cannot model complex logical connections
- **Context fragmentation** - treats passages independently without global reasoning
- **Query reformulation challenges** - complex questions may need rephrasing
```

```{tab-item} Semantic Gaps
- **Paraphrase sensitivity** - may miss semantically equivalent entity references
- **Limited context modeling** - cannot capture long-range dependencies in reasoning
- **Synonym problems** - entity mentions in different forms may be missed
- **Relation understanding** - struggles with implicit relationships between entities
```

```{tab-item} Multihop QA Issues
- **Bridge entity challenges** - may miss intermediate reasoning steps
- **Comparison questions** - struggles with comparative reasoning requirements
- **Evidence aggregation** - no learned mechanism to combine evidence from multiple passages
- **Answer synthesis** - relies on simple generation rather than learned reasoning
```
:::

### Why Chosen Over Alternatives

| Alternative Method | Why Rejected |
|-------------------|--------------|
| **TF-IDF only** | BM25 is more effective for entity retrieval in HotpotQA |
| **Boolean retrieval** | Too rigid for natural language multihop questions |
| **Dense-only retrieval** | Missing exact entity matching capabilities of sparse methods |
| **Traditional QA models** | Cannot handle the multihop reasoning requirements |

## 4.3 Libraries and Frameworks for Implementation

### Primary Framework: HuggingFace + rank-bm25

:::{admonition} Recommended Tech Stack
:class: tip
**Core Libraries:**
- `transformers` - DPR models and Mistral-7B-Instruct
- `rank-bm25` - BM25 implementation for sparse retrieval
- `sentence-transformers` - BGE reranker model
- `datasets` - HotpotQA dataset handling and preprocessing
- `torch` - PyTorch backend for neural components
:::

### Off-the-Shelf Components

#### 1. BM25 Sparse Retrieval
```python
# BM25 implementation using rank-bm25 library
from rank_bm25 import BM25Okapi
import string

# Preprocessing for BM25 on HotpotQA
def preprocess_text(text):
    # Simple tokenization for entity matching
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Initialize BM25 index
corpus_tokens = [preprocess_text(doc) for doc in corpus]
bm25 = BM25Okapi(corpus_tokens)
```

#### 2. DPR Dense Retrieval
```python
# DPR implementation using HuggingFace transformers
from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

# Load pre-trained DPR models
q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
c_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
c_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
```

#### 3. BGE Cross-Encoder Reranker
```python
# BGE cross-encoder reranker for passage reranking
from sentence_transformers import CrossEncoder

# Load pre-trained BGE cross-encoder reranker
reranker = CrossEncoder('BAAI/bge-reranker-base')

# Rerank passages using cross-encoder architecture
# Each (query, passage) pair gets full cross-attention processing
query_passage_pairs = [(query, passage) for passage in top_passages]
scores = reranker.predict(query_passage_pairs)

# Cross-encoder processes each pair through:
# 1. Concatenated input: [CLS] query [SEP] passage [SEP]
# 2. Full cross-attention between all tokens
# 3. Classification head outputs relevance score
```

#### 4. Mistral-7B Answer Generation
```python
# Mistral-7B for answer generation
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Mistral-7B-Instruct
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
```

#### 5. HotpotQA Dataset Processing
```python
# HotpotQA dataset loading and preprocessing
from datasets import load_dataset

# Load HotpotQA dataset
dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor')

# Extract context paragraphs for retrieval corpus
def extract_contexts(examples):
    contexts = []
    for context_list in examples['context']:
        for title, sentences in context_list:
            context = ' '.join(sentences)
            contexts.append(context)
    return contexts
```

:::{note}
**Implementation Advantage:** Pre-trained models provide immediate capability without fine-tuning. The main development work involves HotpotQA dataset preprocessing and evaluation framework integration for multihop reasoning assessment.
:::

## 4.4 Implementation Strategy

### BM25/DPR Retrieval Pipeline
1. **HotpotQA Preprocessing** - Extract passages from context paragraphs
2. **Dual Index Creation** - Build both BM25 sparse index and DPR dense embeddings
3. **Query Processing** - Process questions for both sparse and dense retrieval
4. **Hybrid Retrieval** - Combine BM25 and DPR scores for candidate selection
5. **Top-k Selection** - Return top-100 passages for reranking

### BGE Cross-Encoder Reranking Pipeline
1. **Passage Candidate Input** - Take top-100 from retrieval stage
2. **Query-Passage Pair Formation** - Create input pairs for cross-encoder processing
3. **Cross-Encoder Processing** - Apply BAAI/bge-reranker-base with full cross-attention
   - Concatenate: [CLS] query [SEP] passage [SEP]
   - Process through shared Transformer layers with cross-attention
   - Generate relevance score via classification head
4. **Top-k Reranking** - Select top-10 highest scored passages based on cross-encoder scores
5. **Evidence Preparation** - Format reranked passages for answer generation

### Mistral-7B Generation Pipeline
1. **Context Assembly** - Combine top reranked passages into context
2. **Prompt Construction** - Format question and context for Mistral-7B
3. **Answer Generation** - Use Mistral-7B-Instruct for answer synthesis
4. **Post-processing** - Extract and format final answer
5. **Evaluation** - Assess against HotpotQA metrics (EM, F1, Joint EM)

:::{tip}
**Multihop Optimization**: The pipeline benefits from entity-focused preprocessing, careful prompt engineering for complex questions, and evaluation using HotpotQA's comprehensive metrics including supporting fact identification.
:::

---

```{seealso}
**Next Steps**: These traditional methods establish our baseline performance for multihop reasoning question answering using HotpotQA. We'll compare their hybrid sparse-dense approach against modern end-to-end fine-tuned methods that are specifically optimized for multihop reasoning tasks.

**Evaluation Focus**: Performance will be measured using HotpotQA's comprehensive metrics including Document Recall@k, Supporting-Fact F1, Answer EM/F1, and Joint EM to assess both answer accuracy and reasoning transparency.
```



