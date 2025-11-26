# quantum-string-match

String matching with Grover-style circuits, including ideal simulation, noise sweeps, backend noise emulation, and optional real-hardware runs.

## Setup
- Create a virtual environment and install dependencies:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and fill in your IBM Quantum credentials when using cloud backends.

## Environment variables (examples and effect)
- `QISKIT_IBM_TOKEN` / `QISKIT_IBM_INSTANCE` / `QISKIT_IBM_CHANNEL`  
  - Example: `QISKIT_IBM_CHANNEL=ibm_quantum`  
  - Effect: enables authenticated access to IBM backends; without them hardware runs are skipped.
- `IBM_TARGET_BACKENDS`  
  - Example: `IBM_TARGET_BACKENDS="ibm_kyoto,ibm_osaka"`  
  - Effect: real hardware runs will produce extra summaries labeled `real-ibm_kyoto`, etc.
- `MAX_EXPERIMENTS`  
  - Example: `MAX_EXPERIMENTS=3`  
  - Effect: limits how many of the 25 sentence/pattern pairs are processed to speed up testing.
- `SHOTS`  
  - Example: `SHOTS=1024`  
  - Effect: changes the number of shots per run; higher values reduce sampling noise in `success_prob`.
- `SIM_SEED`  
  - Example: `SIM_SEED=2024`  
  - Effect: fixes transpiler and simulator seeds so simulator results are reproducible.
- `PLOT_HIST`  
  - Example: `PLOT_HIST=0` (no plots) or `PLOT_HIST=1` (show plots)  
  - Effect: toggles histogram rendering; when off you still get CSV summaries.
- `RESULTS_CSV`  
  - Example: `RESULTS_CSV="data/results_summary.csv"`  
  - Effect: location of the consolidated table with overlaps and success probabilities.
- `HISTOGRAM_DIR`  
  - Example: `HISTOGRAM_DIR="data/histograms"`  
  - Effect: directory where histograms are saved (per `experiment_id` subfolder) as `.pgf` files ready for LaTeX inclusion.

## Running
- With plots off (fast):  
  - `PLOT_HIST=0 python src/main.py`
- With plots on:  
  - `PLOT_HIST=1 python src/main.py`
- Hardware runs (requires credentials and target list):  
  - `IBM_TARGET_BACKENDS="ibm_kyoto,ibm_osaka" python src/main.py`

## Outputs
- Console: per-experiment summaries showing expected indices, top indices, overlap count, and `success_prob`.
- CSV: saved to `RESULTS_CSV` (default `data/results_summary.csv`) containing all summaries across ideal, noise sweeps, backend noise emulation, and hardware (if enabled).
- Histograms: saved as `.pgf` under `HISTOGRAM_DIR/<experiment_id>/`, one per run (ideal, backend noise, hardware) and one combined sweep per noise type.
