# Source Code

This directory contains the main source code for DrHouse.

## Setup

**Create the medical database:**
```bash
python src/create_medical_database.py
```

**Train semantic filter:**
```bash
python src/semantic_filter.py
```

## Usage

### Real-world Data Consultation

```bash
python src/main.py --exp real --user_name cj_t2
```

### Simulation Data Consultation

**Two-agent consultation:**
```bash
python src/simulation/simulation_consultation.py
```

**Real person interactions:**
```bash
python src/main.py --exp simulation --user_name sim_data_ab_1
```

### Evaluation

```bash
python src/eval_dialogues.py
```

## User Interfaces

### Web UI

```bash
python src/mainWebUI.py
```

### Android UI

1. Start the server:
```bash
python src/mainAndroidUIServer.py
```

2. Run the Android app: `DrHouse`