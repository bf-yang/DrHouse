# Guideline Tree Retrieval

This module compares retrieval accuracy between MedDM and our mapping-based approach for guideline tree retrieval.

## Methods

### MedDM (Guideline Vector Database)

1. Create the guideline vector database:
```bash
python src/guideline_tree_retrieval/create_guideline_database_meddm.py
```

2. Run MedDM retrieval:
```bash
python src/guideline_tree_retrieval/meddm.py
```

### Mapping-based Approach (Query Transform)

Run the mapping-based retrieval:
```bash
python src/guideline_tree_retrieval/mapping-based.py
```

## Results

### MedDM
- Accuracy: ~70% (depends on RAG split parameters)

### Mapping-based
- Using `Symptom2Disease.csv` as training set: **97.4%**
- Using `Symptom2Disease_train.csv` as training set: **63.7%**
- Overall range: 63.7% ~ 97.4% (depends on training set size)