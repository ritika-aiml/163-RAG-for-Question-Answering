# Evaluation Methods for Multihop Question Answering

## Overview

Evaluating multihop Question Answering (Q&A) systems like those using HotpotQA requires specialized metrics that assess multiple stages of the reasoning pipeline. Unlike single-hop QA systems that extract answers from single passages, multihop QA evaluation must measure:

1. **Document Retrieval**: Finding relevant supporting documents
2. **Evidence Selection**: Identifying key sentences within documents  
3. **Answer Synthesis**: Generating correct answers from multi-document evidence
4. **Joint Performance**: Combined accuracy across all reasoning stages

## HotpotQA-Specific Evaluation Metrics

The HotpotQA dataset requires evaluation at four distinct levels, each measuring different aspects of the multihop reasoning process:

### 1. Document-Level Retrieval: Recall@k (Hits@k)

**Definition**: Measures whether both gold Wikipedia articles are present in the top-k retrieved documents.

**Purpose**: 
- Evaluates the retrieval system's ability to identify relevant supporting documents
- Tests document selection without considering sentence granularity or final answers
- Critical first step in the multihop reasoning pipeline

**Calculation**:
- For each question, check if both gold supporting documents appear in top-k results
- Binary outcome: 1 if both documents present, 0 otherwise
- Typically measured at k=10 (matching HotpotQA's 10-document setting)

**Why chosen**:
- **Prerequisite for success**: Without correct documents, downstream reasoning fails
- **Clear interpretation**: Binary success/failure at document retrieval stage
- **Standard retrieval metric**: Widely used in IR and RAG systems

### 2. Evidence Selection: Supporting-Fact F1

**Definition**: Token-level F1 score measuring overlap between predicted and gold supporting sentences.

**Purpose**:
- Evaluates sentence-level evidence identification within retrieved documents  
- Tests the model's ability to pinpoint specific reasoning-critical sentences
- Measures precision and recall of supporting evidence selection

**Calculation**:
- Precision = (Correct supporting sentences predicted) / (Total sentences predicted)
- Recall = (Correct supporting sentences predicted) / (Total gold supporting sentences)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Why chosen**:
- **Fine-grained evaluation**: Goes beyond document-level to sentence-level precision
- **Balances precision/recall**: Penalizes both missing evidence and over-selection
- **Reasoning transparency**: Measures explainability of the reasoning process

### 3. Answer Synthesis: Answer EM/F1

**Definition**: Standard QA metrics measuring correctness of the final predicted answer.

**Answer Exact Match (EM)**:
- Binary metric: 1 if predicted answer exactly matches ground truth (after normalization), 0 otherwise
- Normalization: lowercase, remove punctuation, strip articles (a/an/the)

**Answer F1**:
- Token-level F1 between predicted and ground truth answers
- Precision = (Shared tokens) / (Predicted answer tokens)
- Recall = (Shared tokens) / (Ground truth tokens)

**Purpose**:
- **Final outcome measurement**: Tests ultimate goal of QA system
- **Answer quality assessment**: EM for strict evaluation, F1 for partial credit
- **Standard QA evaluation**: Comparable to other QA benchmarks

### 4. Joint Exact Match (Joint EM)

**Definition**: Strict, all-or-nothing metric ensuring models both produce correct answers AND cite correct supporting evidence.

**Requirements for Joint EM = 1**:
1. **Answer EM = 1**: Predicted answer exactly matches ground truth
2. **Supporting-Fact EM = 1**: Predicted supporting sentences exactly match gold supporting sentences

**Purpose**:
- **Comprehensive evaluation**: Measures both reasoning correctness and transparency
- **Prevents answer shortcuts**: Ensures models don't bypass proper reasoning
- **Faithfulness requirement**: Verifies that correct answers come from correct evidence

**Why critical for multihop QA**:
- **Reasoning verification**: Confirms models follow intended reasoning paths
- **Explainability enforcement**: Requires models to show their work
- **Real-world applicability**: High-stakes applications need both correct answers and valid reasoning

## Metric Selection Rationale

For HotpotQA multihop reasoning evaluation, we prioritize this four-tier approach because:

1. **Document Recall@k**: Essential first step - no correct documents = guaranteed failure
2. **Supporting-Fact F1**: Measures reasoning quality and explainability 
3. **Answer EM/F1**: Standard QA evaluation for final performance
4. **Joint EM**: Gold standard combining answer correctness with reasoning faithfulness

This combination provides:
- **Pipeline evaluation**: Each stage of the multihop reasoning process
- **Failure point identification**: Pinpoints where system breaks down
- **Comprehensive coverage**: From retrieval through final answer generation
- **Reasoning faithfulness**: Ensures transparency and explainability

## Evaluation Scenarios and Examples

### Excellent Performance Example
**Question**: "Which magazine was started first Arthur's Magazine or First for Women?"

**Gold Data**:
- Supporting documents: ["Arthur's Magazine", "First for Women"]  
- Supporting facts: [("Arthur's Magazine", 0), ("First for Women", 0)]
- Answer: "Arthur's Magazine"

**Model Prediction**:
- Retrieved docs: ["Arthur's Magazine", "First for Women", ...] (top 10)
- Predicted supporting facts: [("Arthur's Magazine", 0), ("First for Women", 0)]
- Predicted answer: "Arthur's Magazine"

**Metrics**:
- **Document Recall@10**: 1.0 ✅ (both gold docs in top-10)
- **Supporting-Fact F1**: 1.0 ✅ (perfect sentence identification)  
- **Answer EM**: 1.0 ✅ (exact match)
- **Joint EM**: 1.0 ✅ (answer + evidence both perfect)

### Partial Success Example
**Question**: "What position did the player drafted by Phoenix play?"

**Gold Data**:
- Supporting documents: ["Player X", "Phoenix Draft"]
- Supporting facts: [("Player X", 2), ("Phoenix Draft", 1)]  
- Answer: "point guard"

**Model Prediction**:
- Retrieved docs: ["Player X", "Phoenix Draft", ...] (correct docs retrieved)
- Predicted supporting facts: [("Player X", 2)] (missed second fact)
- Predicted answer: "point guard"

**Metrics**:
- **Document Recall@10**: 1.0 ✅ (both docs retrieved)
- **Supporting-Fact F1**: 0.67 ⚠️ (1 of 2 supporting facts found)
- **Answer EM**: 1.0 ✅ (correct answer despite incomplete evidence)
- **Joint EM**: 0.0 ❌ (failed due to incomplete supporting facts)

### Poor Performance Example
**Question**: "Which company founded by Elon Musk focuses on space travel?"

**Gold Data**:
- Supporting documents: ["Elon Musk", "SpaceX"]
- Supporting facts: [("Elon Musk", 3), ("SpaceX", 0)]
- Answer: "SpaceX"

**Model Prediction**:
- Retrieved docs: ["Elon Musk", "Tesla", "Neuralink", ...] (missed SpaceX)
- Predicted supporting facts: [("Tesla", 1)] (wrong evidence)
- Predicted answer: "Tesla"

**Metrics**:
- **Document Recall@10**: 0.0 ❌ (missed critical SpaceX document)
- **Supporting-Fact F1**: 0.0 ❌ (no overlap with gold supporting facts)
- **Answer EM**: 0.0 ❌ (incorrect answer)
- **Joint EM**: 0.0 ❌ (complete failure)

## Limitations and Challenges

### Metric Limitations

1. **Supporting-Fact F1 Strictness**:
   - May penalize semantically equivalent but differently phrased evidence
   - Sentence boundaries can be arbitrary for reasoning chains

2. **Answer EM Harshness**:
   - "twenty-five percent" vs "25%" → scored as incorrect
   - Doesn't handle multiple valid answer formulations

3. **Joint EM All-or-Nothing**:
   - Single supporting fact error → complete failure
   - May be too strict for real-world applications

4. **Document Recall Limitations**:
   - Doesn't measure ranking quality within top-k
   - Binary success/failure ignores near-misses

### Scenarios Where Metrics May Struggle

1. **Paraphrase Equivalence**:
   - Gold: "established in 1844"
   - Prediction: "founded in 1844" → May be penalized despite correctness

2. **Reasoning Chain Variations**:
   - Multiple valid paths to same answer
   - Different but equally valid supporting evidence

3. **Temporal Answer Variations**:
   - "January 1, 2020" vs "Jan 1 2020" vs "01/01/2020"

## Implementation Best Practices

1. **Comprehensive Evaluation**: Report all four metrics for complete system assessment
2. **Error Analysis**: Identify failure points (retrieval vs reasoning vs synthesis)  
3. **Baseline Comparison**: Compare against retrieval baselines and prior work
4. **Ablation Studies**: Test individual components to understand contributions
5. **Human Evaluation**: Supplement with human judgment for edge cases

```{tip}
The HotpotQA evaluation framework provides comprehensive assessment of multihop reasoning systems by evaluating each stage of the pipeline. The four-metric approach ensures systems must succeed at document retrieval, evidence selection, answer synthesis, and reasoning faithfulness to achieve high Joint EM scores.
```
