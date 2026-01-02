# Real-world Experiments

This directory contains code and data for real-world experiments.

## Usage

1. **Generate vector database for sensor data:**
```bash
python create_sensordata_vector_db.py
```

2. **Run real-world consultation:**
```bash
python src/main.py --exp real --user_name xxx
```

## Directory Structure

- `data/`: Real-world profiles used for experiments
- `results/`: Generated dialogues and evaluation scores