# Cloud control-plane trace analytics

Simulation and analysis of configuration error states from cloud control-plane events.
All learners and cloud accounts in the supplied data are simulated.

## Setup

Use Python 3.11 and install the tested dependencies:

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements-lock.txt
```

Run commands from this directory. On Windows, set `OMP_NUM_THREADS=1` before the analysis commands.

## Tests and figures

```sh
python -m pytest -q tests
python make_figures2.py
```

Figures 1–3 are written to `tex/figures/` as PDF and PNG. The figure script reads supplied analysis results; it does not rerun the simulation.

| Figure | Inputs in `results/` | Output stem |
|---|---|---|
| 1 | Framework defined in the figure script | `fig1_framework` |
| 2 | `numbers.json`, `new_part1.json`, `loss_all.json` | `fig2_channels` |
| 3 | `rq4_results.json` | `fig3_designs` |

## Data

- `data/public/corpus_R6.jsonl.gz`: 213,574 events from 2,000 simulated learners. Time is represented by session order and within-session seconds.
- `data/public/ground_truth.jsonl.gz`: one record per simulated learner, including labels and latent variables; course timetable fields are removed.
- `results/`: supplied analysis outputs used for the manuscript figures and tables.
- `rq4/rq4_inputs.npz`: derived learner-level features and outcomes used by the RQ4 analysis, including synthetic simulator identifiers.

The event and ground-truth files join on `learner` and event sequence number `seq`. The R6 release omits absolute timestamps and between-session gaps. The original analysis drivers expect raw simulator records, not the R6 release.

## Simulation and analysis

```sh
python run_all.py simulate
python run_all.py analyse
python run_all.py revision
python make_figures2.py
```

Simulation writes 16 shards to `sim/main/`, with 2,000 learners and per-learner seeds `SeedSequence([20261104, i])`. Revision analyses generate the state-dump cohort and sensitivity variants and run the event-loss, strategy, HMM, privacy and RQ4 studies. These commands are substantially more expensive than tests or figure generation. Raw shards are generated locally and are not included in this repository.

RQ4 can also run separately:

```sh
cd rq4
python rq4_run.py rq4_inputs.npz all
```

Copy the resulting `rq4_results.json` to `results/` before regenerating Figure 3. `build_rq4_inputs_cell.py` records the original interactive extraction and needs its documented analysis-session objects; it is not a standalone extraction command. The supplied NPZ provides the input for rerunning RQ4.

To export newly simulated raw shards in R6 format:

```sh
python build_release.py data/generated_R6
```

This creates a new random pseudonym key; learner pseudonyms will differ from the supplied release.

## License

Code is licensed under the [MIT License](LICENSE). The simulated data in `data/public/`, derived inputs in `rq4/rq4_inputs.npz`, and precomputed results in `results/` are licensed under [CC BY 4.0](LICENSE-DATA). For attribution, cite this repository and the data filename.
