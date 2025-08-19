# Data Description

## Dataset Overview

For this multi-hop reasoning question answering project, we focus on the **HotpotQA dataset**, a large-scale dataset specifically designed for multi-hop reasoning over text. This dataset was selected because it directly addresses the core challenge of our project: answering questions that require synthesizing information from multiple sources rather than extracting answers from single passages.

## Dataset Creation Methodology

### Data Generation Process

HotpotQA was developed by researchers from Carnegie Mellon University, Stanford University, and Université de Montréal, following a systematic crowdsourcing approach:

- **Source**: Available on Hugging Face at `hotpotqa/hotpot_qa` (distractor setting)
- **Crowdsourcing Framework**: Human annotators (crowdworkers) were presented with pairs of Wikipedia introduction paragraphs and asked to formulate questions requiring information from both articles
- **Supporting Facts Annotation**: Each question includes sentence-level supporting facts that crowdworkers identify as necessary for reasoning, enabling strong supervision and explainable predictions
- **Quality Control**: Questions were validated to ensure genuine multi-hop reasoning requirements, avoiding trivial single-paragraph solutions
- **Wikipedia Base**: All supporting contexts sourced from English Wikipedia introduction paragraphs, providing factual and encyclopedic knowledge

### Question Type Design

The dataset incorporates diverse reasoning strategies:
- **Bridge Entity Questions**: Require finding intermediate entities to connect question to answer
- **Comparison Questions**: Involve comparing two entities by a common attribute
- **Missing Entity Questions**: Where key entities are not explicitly mentioned in the question
- **Intersection Questions**: Testing ability to find entities satisfying multiple properties

*Reference: Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.*

## Dataset Structure and Instances

The HotpotQA dataset contains the following key components:

- **Training Set**: ~90,000 question-answer pairs
- **Validation Set**: ~7,400 question-answer pairs
- **Test Set**: Available but labels withheld for evaluation

### Key Fields:
- **question**: The multi-hop question requiring reasoning across contexts
- **answer**: The ground truth answer (typically entity names or short phrases)
- **context**: List of 10 paragraphs (2 gold + 8 distractors) from Wikipedia articles
- **supporting_facts**: Annotated evidence sentences that support the answer
- **type**: Question classification (comparison, bridge entity questions)
- **level**: Difficulty level (easy, medium, hard)

## Dataset Characteristics

### Question Types Distribution:
- **Bridge Questions** (~80%): Require finding an intermediate entity to bridge to the final answer
- **Comparison Questions** (~20%): Require comparing properties of two or more entities

### Complexity Metrics:
- **Average Question Length**: ~15-20 words per question
- **Answer Length**: Typically 1-3 words (entity names, dates, numbers)
- **Supporting Facts**: Usually 2-4 evidence sentences per question
- **Context Length**: Each example contains ~2,000-3,000 words across 10 paragraphs

## Data Quality and Challenges

### Strengths:
- **Genuine Multi-hop Nature**: Questions cannot be answered using single passages
- **Diverse Topics**: Covers wide range of Wikipedia knowledge domains
- **Annotated Evidence**: Supporting facts provide supervision for reasoning chains
- **Balanced Difficulty**: Questions span different complexity levels

### Potential Issues:
- **Annotation Consistency**: Some supporting facts may be subjectively chosen
- **Context Noise**: 8 out of 10 paragraphs are distractors, creating realistic but challenging scenarios
- **Answer Type Bias**: Heavily skewed toward named entity answers
- **Wikipedia Limitations**: Knowledge bounded by Wikipedia's coverage and recency

## Evaluation Methodology

### Evaluation Settings

HotpotQA provides two distinct evaluation configurations that test different aspects of multi-hop reasoning systems:

#### 1. Distractor Setting (Closed Book + Retrieval Challenge)
This setting provides:
- **A set of 10 Wikipedia paragraphs for each question**
- **One of these includes the gold supporting facts** (i.e., evidence sentences)
- **The others are distractor paragraphs** — plausible but irrelevant

Your model must:
- Pick relevant paragraphs
- Find supporting sentences
- Generate the correct answer

👀 **Use case**: Evaluates both retrieval and reasoning capabilities in a controlled environment

#### 2. Full Wiki Setting (Open Domain QA)
This setting does not give you the paragraphs — instead, you must:
- **Retrieve paragraphs yourself from the full Wikipedia dump**
- **Then reason across them**

👀 **Use case**: End-to-end QA pipeline (retrieval + reasoning) testing complete system performance

### Evaluation Metrics

The dataset employs comprehensive evaluation measures:

- **Answer Accuracy**: 
  - Exact Match (EM): Strict string matching between predicted and gold answers
  - F1 Score: Token-level overlap between predicted and gold answers
- **Supporting Facts Evaluation**: 
  - Precision/Recall/F1 for identifying correct supporting sentences
  - Joint evaluation combining answer accuracy and supporting fact identification
- **Explainability Assessment**: Models evaluated on both correctness and reasoning transparency

### Evaluation Platform

- Test set evaluation conducted through Codalab
- Distractor setting requires code submission with Docker environment
- Fullwiki setting accepts prediction file submissions

## Applications and Novel Contributions

### Key Applications

1. **Explainable AI Development**: Enables training of systems that provide transparent reasoning chains
2. **Multi-hop Reasoning Research**: Benchmark for advancing complex reasoning capabilities
3. **Retrieval-Augmented Generation**: Ideal testbed for RAG systems requiring multi-document synthesis
4. **Knowledge Integration**: Testing ground for combining information across diverse sources

### Dataset Novelty

HotpotQA introduces several innovative aspects to QA research:

1. **Sentence-level Supervision**: First large-scale dataset providing explicit supporting fact annotations
2. **Genuine Multi-hop Design**: Systematic validation ensuring questions cannot be answered from single passages  
3. **Diverse Reasoning Types**: Comprehensive coverage of different multi-hop reasoning patterns
4. **Explainability Focus**: Emphasis on interpretable and justifiable question answering
5. **Dual Evaluation Settings**: Both closed-domain and open-domain evaluation paradigms

## Relevance to Multi-hop Reasoning

This dataset is particularly well-suited for our RAG-based approach because:

1. **Multi-document Nature**: Forces retrieval systems to gather information from multiple sources
2. **Reasoning Requirement**: Simple extraction is insufficient; synthesis and inference are required  
3. **Real-world Complexity**: Mimics scenarios where answers require connecting disparate pieces of information
4. **Comprehensive Evaluation**: Provides both answer accuracy and supporting evidence evaluation
5. **Explainability Framework**: Supports development of transparent reasoning systems

The HotpotQA dataset serves as an ideal testbed for developing and evaluating RAG systems capable of sophisticated multi-hop reasoning in question answering tasks.