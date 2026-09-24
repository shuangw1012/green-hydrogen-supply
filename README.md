# Green Hydrogen Supply Optimisation

This repository contains the Python, PySAM and MiniZinc/Gurobi workflow used to
optimise the spatial configuration and hourly operation of a renewable hydrogen
hub. The included dataset reproduces the northern Tasmania demonstration case.

The workflow combines:

- spatial candidate-cell data and interconnection distances;
- hourly PV and wind resource profiles;
- PySAM reference generation models;
- a mixed-integer linear programming (MILP) formulation for capacity sizing,
  siting and hourly dispatch; and
- LCOH and component-cost reporting.

The current MiniZinc model is the revised MILP formulation. PV and wind siting
binaries have been removed because zero continuous capacity already represents
an undeveloped cell. The remaining products between the binary electrolyser
location and bounded continuous infrastructure-cost variables are represented
with exact binary-continuous linearisation constraints.

## Repository structure

```text
.
├── DATA/
│   ├── OPT_OUTPUTS/                  # generated results (ignored by Git)
│   └── SAM_INPUTS/
│       ├── SOLAR/                    # PySAM PVWatts configuration
│       ├── WEATHER_DATA/             # Tasmania 2018 weather inputs
│       └── WIND/
│           ├── SAM_results/          # cached 2018 wind generation profiles
│           └── windfarm_windpower.json
├── MINIZINC/
│   └── hydrogen_plant_MILP.mzn       # optimisation model
├── PYTHON/
│   ├── Input_Tas.txt                 # candidate cells, user and storage site
│   ├── input.txt                     # scenario table
│   ├── main.py                       # scenario runner / MPI dispatcher
│   ├── projdirs.py                   # project-relative paths
│   └── PACKAGE/
│       ├── component_model.py        # PySAM resource and generation models
│       └── optimisation.py           # .dzn generation and solver interface
├── scripts/
│   └── run_nci_17cpus.pbs           # example NCI/Gadi PBS script
└── requirements.txt
```

## Software requirements

Python dependencies are listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The workflow also requires:

1. **MiniZinc** available on `PATH` as `minizinc`;
2. **Gurobi** available to MiniZinc, with a valid licence; and
3. **NREL PySAM** (`NREL-PySAM` Python package).

`mpi4py` is only required for parallel scenario execution. If it is not
installed, `PYTHON/main.py` falls back to serial execution.

## Run the Tasmania scenarios locally

From the repository root:

```bash
python PYTHON/main.py
```

Each row of `PYTHON/input.txt` is one independent scenario. Results are written
to `DATA/OPT_OUTPUTS/`.

The code uses paths relative to the repository itself, so it does not depend on
the shell working directory.

## Run scenarios in parallel on NCI/Gadi

The included PBS template requests 17 CPUs:

```bash
qsub scripts/run_nci_17cpus.pbs
```

Before submission, adapt the project code, memory/walltime request, and module
or virtual-environment commands to the software environment used on Gadi.

Scenario rows are distributed cyclically across MPI ranks. For the supplied 17
rows and 17 MPI ranks, each rank solves one scenario. Each MiniZinc/Gurobi solve
uses one thread to avoid CPU oversubscription.

## Inputs

### `PYTHON/input.txt`

Scenario-level techno-economic assumptions. Important columns include PV, wind
and electrolyser CAPEX; fixed O&M; storage, transmission, pipeline and battery
cost multipliers; storage technology; weather-data year; hydrogen load; supply
capacity factor; and cost year.

### `PYTHON/Input_Tas.txt`

Spatial inputs for the Tasmania case. The candidate grid cells are followed by
an end-user row (`User`) and an underground-storage row (`storage`). A storage
coordinate of `(0, 0)` is the existing convention for cases without a spatially
constrained underground-storage location.

### Weather and PySAM data

The repository includes the 2018 weather data used by the demonstration case.
PV generation is evaluated with PySAM PVWatts. Wind generation uses PySAM
Windpower; cached 2018 reference profiles are included to avoid repeated wind
simulations.

## MILP formulation

The principal spatial decisions are:

- continuous PV capacity in each candidate cell;
- continuous wind capacity in each candidate cell; and
- one binary variable per candidate electrolyser location.

PV and wind capacity variables can take the value zero, so separate PV/wind
build binaries are unnecessary.

For a bounded continuous infrastructure cost `C` and an electrolyser-location
binary `y`, the original bilinear term `C*y` is represented by an auxiliary
variable `z` and the exact linear constraints

```text
z <= C
z <= U*y
z >= C - U*(1-y)
z >= 0
```

where `U` is a valid upper bound. The MiniZinc model applies this reformulation
to the location-dependent transmission and pipeline cost terms. The output
variable names are retained for compatibility with the Python parser.

## Generated files

During a run, each process creates temporary resource and MiniZinc data files
using a unique run identifier. They are removed after the scenario finishes and
are excluded by `.gitignore`.

Generated summary and hourly result CSVs are written to `DATA/OPT_OUTPUTS/` and
are also excluded from version control.

## Code provenance

The original green-hydrogen supply modelling code was primarily developed by
Ahmad Mojiri for HILT CRC project RP2.001. The present repository was adapted
and further developed for the GIS-integrated renewable hydrogen hub study by
Shuang Wang and collaborators.

## Reproducibility note

The cleaned repository intentionally preserves the optimisation equations and
output keys used by the current study. A separate `CLEANUP_NOTES.md` records
code-cleaning changes and a small number of modelling details that should be
reviewed before the repository is archived with a publication.
