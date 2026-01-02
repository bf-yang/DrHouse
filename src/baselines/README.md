# Baselines

This module contains baseline implementations for comparison with DrHouse.

## GPT-3.5 / GPT-4 Baseline

Uses the same interface as `python src/main.py` but with baseline models.

**Real-world experiments:**
```bash
python src/baselines/main_baseline.py --model_version gpt-4-1106 --exp real --user_name cj_t2
```

**Simulation experiments:**
```bash
python src/baselines/main_baseline.py --model_version gpt-4-1106 --exp simulation --user_name sim_data_ab_1
```

## MedDM Baseline

**Setup:**
1. Generate vector database for guideline trees:
```bash
python src/baselines/create_guideline_vector_db.py
```

2. Run MedDM baseline:
```bash
python src/baselines/main_meddm.py
```