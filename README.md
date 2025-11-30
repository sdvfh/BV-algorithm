# Bernstein-Vazirani algorithm

Bernstein-Vazirani experiments executed across ideal simulation, backend-noise emulation, custom noise injections, and optional IBM Quantum hardware, with automated plotting and metric aggregation.

## Algorithm
- Implements the Bernstein-Vazirani oracle and circuit for 5-qubit secrets (one ancillary qubit for phase kickback).
- Evaluates a fixed suite of 25 secret bitstrings to benchmark robustness under varied noise conditions.
- Single-shot oracle query recovers the hidden bitstring; correctness is assessed via the most frequent measurement outcome.

## Experiment flow
- Ideal simulation: Aer simulator without noise for each secret.
- Backend-noise emulation: Aer simulator with noise models derived from each available IBM backend (excluding `ibm_marrakesh`).
- Real hardware (optional): runtime Sampler execution on each available backend; failures are logged and skipped.
- Readout sweeps: injected symmetric readout errors across predefined levels (`ultra-low` to `very-high`) with ideal overlays.
- Custom noise sweeps: depolarizing, amplitude/phase damping, thermal relaxation, and gate-specific noise across the same intensity levels.
- Plots: horizontal histograms comparing ideal, emulated, noisy, and real results; readout and custom sweeps are saved per secret and noise kind.

## Setup
- Create an isolated Python environment and install dependencies:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Populate `.env` from `.env.example` when using cloud backends.

## Environment variables
- `QISKIT_IBM_TOKEN`  
  - IBM Quantum API token used by `QiskitRuntimeService` for authentication.
- `QISKIT_IBM_INSTANCE`  
  - IBM Cloud instance CRN to scope backend access.
- `QISKIT_IBM_CHANNEL`
  - Service channel passed to `QiskitRuntimeService`; if omitted, default channel resolution is used.

## Running
- Execute the Bernstein-Vazirani suite:  
  - `python src/bv.py`
- The script auto-discovers available IBM backends (excluding `ibm_marrakesh`) and attempts real execution when credentials permit.

## Outputs
- Predictions: `Plot_results/bv_predictions.csv` with per-run category, backend, expected secret, predicted bitstring, correctness flag, top outcome probability, and raw counts.
- Metrics: `Plot_results/bv_metrics.csv` with accuracy, macro precision/recall, and macro F1 per `(category, backend)` pair, including noise annotations.
- Histograms: PNG files under `Plot_results/` grouped by category:
  - `ideal/` and `noisy/` histograms per secret and backend.
  - `backend_comparison/` overlays of ideal vs emulated vs real (when available).
  - `custom_noise/readout/` sweep plots per secret; analogous folders for each custom noise kind.
- Cached counts: JSON histograms under `Plot_results/hist_data/` to avoid recomputation on subsequent runs.
