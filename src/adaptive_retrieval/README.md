# Adaptive Retrieval

This module implements adaptive retrieval for sensor data using LLM-based methods.

## Dataset

The `data/` folder contains our collected datasets for adaptive retrieval experiments.

## Experiments

### Semantic Filter

Train and evaluate the semantic filter model:

```bash
python src/adaptive_retrieval/semantic_filter.py
```

### LLM-based Retrieval

**Few-shot learning:**
```bash
python src/adaptive_retrieval/llm_fewshot.py
```

**Zero-shot inference:**
```bash
python src/adaptive_retrieval/llm_zeroshot.py
```