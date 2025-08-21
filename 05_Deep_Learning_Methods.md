# Deep Learning RAG Pipeline Training Plan for HotpotQA

## 1. Goals & Scope

**Primary Goal**: Maximize HotpotQA Answer EM/F1 while maintaining high faithfulness (answers supported by cited passages).

**Secondary Goals**: Strong Supporting-Facts F1, robust to distractors, stable latency under 2s.

**Architecture**: Frozen dense retriever → Fine-tuned cross-encoder reranker → QLoRA-tuned Mistral-7B generator.

## 2. Colab-Friendly Adjustments

### Hardware Considerations
- **T4 GPU (Free Colab)**: 16GB VRAM, sufficient for QLoRA training
- **A100/L4 (Colab Pro)**: Can handle larger batch sizes and sequence lengths
- **Session Limits**: 12-hour maximum, requiring robust checkpointing

### Memory Optimizations
- **Re-ranker**: Small cross-encoder (100-400M params) can be fully fine-tuned on T4
- **Generator**: Mistral-7B with 4-bit quantization + QLoRA adapters only
- **Sequence Length**: ≤2048 tokens on T4, ≤4096 on A100
- **Training Schedule**: 1-2 epochs max to fit session limits

## 3. Data & Preprocessing

### Training Bundle Construction
- **Retrieve**: 200 candidates per question using frozen BGE-large
- **Re-rank**: Keep top k=8-12 passages after reranking
- **Chunking**: 150-250 tokens per passage, preserve title context

### Curriculum Learning Strategy
```python
# Epochs 1-2: Force curriculum
curriculum_passages = gold_pages + hard_negatives + retrieval_candidates[:remaining]

# Epochs 3+: Pure retrieval 
realistic_passages = retrieval_candidates[:k]  # Gold may be missing
```

### Target Format
```json
{
  "question": "Which magazine started first?",
  "passages": [
    {"id": 0, "title": "Arthur's Magazine", "text": "Founded in 1844..."},
    {"id": 1, "title": "First for Women", "text": "Launched in 1989..."}
  ],
  "target": "<answer>Arthur's Magazine [0]",
  "supporting_facts": [("Arthur's Magazine", 0), ("First for Women", 1)]
}
```

## 4. Re-ranker Training (Stage 1)

### Colab-Optimized Configuration
```yaml
model: cross-encoder/ms-marco-MiniLM-L-6-v2
max_length: 256
learning_rate: 2e-5
batch_size: 32  # T4-friendly
epochs: 2       # Reduced for session limits
fp16: true
gradient_checkpointing: true
```

### Training Data
- **Positives**: Passages containing supporting sentences from gold facts
- **Hard negatives**: High-retrieval-score passages without supporting facts
- **Loss**: Pairwise margin loss or softmax over candidates

## 5. Generator QLoRA Training (Stage 2)

### Colab-Optimized Model Configuration
```yaml
base_model: mistralai/Mistral-7B-Instruct-v0.2
quantization: 4bit-nf4
lora_rank: 16        # Reduced for memory
lora_alpha: 32
lora_targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
gradient_checkpointing: true
```

### Input Template
```python
template = """[Question]
{question}

[Evidence]
[1] {title_1}: {passage_1}
[2] {title_2}: {passage_2}
...
[{k}] {title_k}: {passage_k}

[Instruction]
Answer concisely using the evidence. If unsure, say "insufficient context".
Respond with: <answer> and cite indices like [1], [3].

<answer>"""
```

### Colab Training Configuration
```yaml
learning_rate: 5e-4
weight_decay: 0.01
sequence_length: 2048           # T4-safe
batch_size: 1                   # + gradient_accumulation_steps: 8
epochs: 2                       # Reduced for session limits
gradient_checkpointing: true
bf16: true                      # Use bf16 if available
save_steps: 100                 # Frequent saves
eval_steps: 100                 # Regular evaluation
```

## 6. Weights & Biases Integration

### Checkpoint Management
- **Adapter-only saves**: Never upload full base model (>13GB)
- **Compressed artifacts**: Zip adapter folders, keep <500MB
- **Aliases**: `latest` (most recent), `best` (highest eval score)
- **Resume capability**: Restore from latest artifact on session restart

### Artifact Structure
```python
artifact_metadata = {
    "step": trainer.state.global_step,
    "eval_f1": eval_results["eval_f1"],
    "eval_em": eval_results["eval_em"], 
    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "lora_rank": 16,
    "sequence_length": 2048
}
```

## 7. Evaluation Protocol

### Core Metrics
- **Answer EM/F1**: Standard HotpotQA answer evaluation
- **Supporting-Facts F1**: Accuracy of evidence identification
- **Supported-Answer Rate**: % answers with correct citations ≥90%
- **Training Efficiency**: Steps/second, VRAM usage

### Colab-Specific Monitoring
- **Session Time**: Track remaining time, save before timeout
- **Memory Usage**: Monitor VRAM to prevent OOM crashes
- **Checkpoint Size**: Ensure artifacts stay under 500MB limit

## 8. Implementation Strategy

### Notebook Structure
1. **Setup & Detection**: GPU type, VRAM, Drive mounting
2. **W&B Integration**: Login, project setup, artifact management
3. **Data Pipeline**: HotpotQA loading with prompt templates
4. **Model Setup**: Quantized Mistral + QLoRA configuration
5. **Training Loop**: Custom callbacks for artifact management
6. **Evaluation**: Metrics computation and W&B logging
7. **Inference Demo**: Load best model and test generation

### Session Management
```python
# Check remaining session time
def get_session_time_left():
    # Implementation to estimate remaining Colab time
    pass

# Save checkpoint before timeout
if get_session_time_left() < 30:  # 30 minutes left
    trainer.save_model()
    upload_artifact_checkpoint()
```

## 9. Deliverables

### Primary Notebook
- `05_Deep_Learning_Methods_Code.ipynb`: Complete Colab-ready implementation

### Key Features
- **One-click setup**: Install all dependencies, detect hardware
- **Robust checkpointing**: Resume from any point, artifact management
- **Memory optimization**: 4-bit quantization, gradient checkpointing
- **Real-time monitoring**: W&B dashboard, metrics tracking
- **Production inference**: Load best model, generate answers

## 10. Success Criteria

### Technical Milestones
- **Memory Efficiency**: Successful training on T4 (16GB VRAM)
- **Checkpoint Reliability**: Resume from artifacts without issues
- **Artifact Management**: All uploads <500MB, proper versioning

### Performance Targets
- **Answer F1**: >5% improvement over baseline
- **Checkpoint Frequency**: Every 100 steps without session timeout
- **Training Speed**: >50 tokens/second on T4

### Production Readiness
- **Artifact Deployment**: Load best model from W&B in inference
- **Scalability**: Configuration adapts to available hardware
- **Reproducibility**: Fixed seeds, deterministic training

This plan ensures robust, Colab-friendly training while maintaining the core deep learning pipeline objectives for HotpotQA multihop reasoning.