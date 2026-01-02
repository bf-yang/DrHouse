# Simulation Experiments

This directory contains code and data for simulation experiments.

## Usage

1. **Generate vector database for sensor data:**
```bash
python create_sensordata_vector_db.py
```

2. **Run simulation consultation:**
```bash
python src/main.py --exp simulation --user_name sim_data_ab_1
```

## Directory Structure

- `data/`: Real-world profiles used for simulation
- `results/`: Generated dialogues and evaluation scores