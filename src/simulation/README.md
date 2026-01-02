# Simulation

This folder provides code for simulation experiments.

## Modes

### Real Person with Simulated Data

A real person interacts with DrHouse using simulated sensor data and symptoms from public medical datasets. Symptoms are manually input according to the medical datasets.

```bash
python main.py
```

Note: This `main.py` is the same as `src/main.py`.

### LLM Agent as Simulated Patient

An LLM agent role-plays as a simulated patient to interact with DrHouse.

```bash
python src/simulation/simulation_consultation.py
```

## Prompts

- `prompt/data_generation`: Generates simulated sensor data
- `prompt/patient_simulation`: LLM role-play prompts for simulated patients

## Simulated Patient Demo

### CLI Mode

Run simulated patient in command-line interface:

```bash
python src/simulation/patient_simulation.py
```

### Gradio Web UI

**Without patient type selection:**
```bash
python src/simulation/patient_simulationWebUI.py
```

**With patient type selection:**
```bash
python src/simulation/patient_simulationWebUISelect.py
```
