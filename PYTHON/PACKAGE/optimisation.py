"""Optimisation workflow for the renewable-hydrogen hub model.

The module performs three tasks:
1. generate hourly PV and wind reference profiles with PySAM;
2. write a scenario-specific MiniZinc data file; and
3. solve the MILP with the MiniZinc Gurobi backend.

The public function :func:`Optimise` retains the interface used by the original
project scripts.
"""

from __future__ import annotations

import subprocess

import numpy as np

from projdirs import MINIZINC_DIR
from PACKAGE.component_model import (
    SolarResource,
    WindSource_windlab,
    pv_gen,
    wind_gen,
)

MODEL_FILE = MINIZINC_DIR / "hydrogen_plant_MILP.mzn"
PROJECT_LIFETIME_YEARS = 25
PV_REFERENCE_CAPACITY_KW = 1_000.0
WIND_REFERENCE_CAPACITY_KW = 320_000.0
GUROBI_RELATIVE_GAP = 0.001


def _as_python_list(values) -> list:
    """Convert NumPy/pandas numeric values to plain Python values for .dzn."""
    return np.asarray(values).tolist()


def _write_minizinc_matrix(stream, name: str, matrix) -> None:
    """Write a two-dimensional numeric array using MiniZinc matrix syntax."""
    array = np.asarray(matrix)
    stream.write(f"{name} = [")
    for row_index, row in enumerate(array):
        stream.write("|")
        stream.write(", ".join(str(float(value)) for value in row))
        if row_index < len(array) - 1:
            stream.write("\n")
    stream.write("|];\n\n")


def make_dzn_file(
    DT,
    EL_ETA,
    BAT_ETA_in,
    BAT_ETA_out,
    C_PV,
    C_WIND,
    C_EL,
    C_UG_STORAGE,
    UG_STORAGE_CAPA_MAX,
    C_PIPE_STORAGE,
    PIPE_STORAGE_CAPA_MIN,
    C_BAT_ENERGY,
    C_BAT_POWER,
    OM_PV,
    OM_WIND,
    OM_EL,
    OM_UG,
    DIS_RATE,
    CF,
    PV_REF,
    WIND_REF,
    LOAD,
    PV_REF_POUT,
    WIND_REF_POUT,
    Area,
    distancePV,
    distanceWind,
    distanceUser,
    distanceStg,
    C_stg_ratio,
    C_trans_ratio,
    C_pipe_ratio,
    storage_type,
    random,
):
    """Write the MiniZinc data file for one optimisation scenario.

    ``C_stg_ratio`` and ``storage_type`` are retained in the function signature
    for backward compatibility with the scenario dictionary, although their
    effects have already been applied before this function is called.
    """
    del C_stg_ratio, storage_type  # already incorporated upstream

    crf = DIS_RATE * (1 + DIS_RATE) ** PROJECT_LIFETIME_YEARS / (
        (1 + DIS_RATE) ** PROJECT_LIFETIME_YEARS - 1
    )
    h_total = (CF / 100.0) * sum(LOAD) * DT * 3600.0

    capacity_levels = [0, 116000, 579000, 2894000, 5787000]
    trans_capex_values = (
        np.array([0, 826695, 1211205, 2018674, 2556988], dtype=float)
        * C_trans_ratio
    ).tolist()
    pipe_capex_values = (
        np.array([0, 314691, 646349, 1539750, 2258811], dtype=float)
        * C_pipe_ratio
    ).tolist()

    trans_opex_ratio = 0.005 * C_trans_ratio
    pipe_opex_ratio = 0.0225 * C_pipe_ratio

    data_file = MINIZINC_DIR / f"hydrogen_plant_data_{random}.dzn"
    load_list = [float(value) for value in LOAD]

    with data_file.open("w", encoding="utf-8", newline="\n") as stream:
        lines = [
            f"N = {len(load_list)};",
            f"n_PV = {len(PV_REF_POUT)};",
            f"n_wind = {len(WIND_REF_POUT)};",
            f"DT = {float(DT):.2f};",
            f"n_project = {PROJECT_LIFETIME_YEARS};",
            f"EL_ETA = {float(EL_ETA):.2f};",
            f"BAT_ETA_in = {float(BAT_ETA_in):.2f};",
            f"BAT_ETA_out = {float(BAT_ETA_out):.2f};",
            "",
            f"C_PV = {float(C_PV):.2f};",
            f"C_WIND = {float(C_WIND):.2f};",
            f"C_EL = {float(C_EL):.2f};",
            f"C_UG_STORAGE = {float(C_UG_STORAGE):.2f};",
            f"UG_STORAGE_CAPA_MAX = {float(UG_STORAGE_CAPA_MAX):.2f};",
            f"C_PIPE_STORAGE = {float(C_PIPE_STORAGE):.2f};",
            f"PIPE_STORAGE_CAPA_MIN = {float(PIPE_STORAGE_CAPA_MIN):.2f};",
            f"C_BAT_ENERGY = {float(C_BAT_ENERGY):.2f};",
            f"C_BAT_POWER = {float(C_BAT_POWER):.2f};",
            "",
            f"OM_PV = {float(OM_PV):.2f};",
            f"OM_WIND = {float(OM_WIND):.2f};",
            f"OM_EL = {float(OM_EL):.2f};",
            f"OM_UG = {float(OM_UG):.2f};",
            "",
            f"RES_H_CAPA = {(1 - CF / 100.0) * sum(load_list) * DT * 3600.0:.2f};",
            f"PV_REF = {float(PV_REF):.2f};",
            f"WIND_REF = {float(WIND_REF):.2f};",
            f"LOAD = {load_list};",
            f"DIS_RATE = {float(DIS_RATE)};",
            f"crf = {float(crf)};",
            f"H_total = {float(h_total)};",
            f"Area = {_as_python_list(Area)};",
            f"capacityLevels = {capacity_levels};",
            f"TranscapexValues = {trans_capex_values};",
            f"PipecapexValues = {pipe_capex_values};",
            f"TransopexRatio = {float(trans_opex_ratio)};",
            f"PipeopexRatio = {float(pipe_opex_ratio)};",
            f"distanceUser = {_as_python_list(distanceUser)};",
            f"distanceStg = {_as_python_list(distanceStg)};",
            "",
        ]
        stream.write("\n".join(lines))

        _write_minizinc_matrix(stream, "PV_REF_POUT", PV_REF_POUT)
        _write_minizinc_matrix(stream, "WIND_REF_POUT", WIND_REF_POUT)
        _write_minizinc_matrix(stream, "distancePV", distancePV)
        _write_minizinc_matrix(stream, "distanceWind", distanceWind)

    return data_file


def _parse_minizinc_output(output: str) -> dict[str, np.ndarray]:
    """Parse the deliberately simple ``key=value;`` MiniZinc output format."""
    result_block = None
    for block in output.split("!"):
        if "CAPEX=" in block:
            result_block = block
            break

    if result_block is None:
        raise RuntimeError("MiniZinc returned no parseable optimisation result.")

    results: dict[str, np.ndarray] = {}
    for item in result_block.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        value = value.strip().strip("[]")
        if value:
            results[key.strip()] = np.fromstring(value, sep=",", dtype=float)
        else:
            results[key.strip()] = np.array([], dtype=float)
    return results


def Minizinc(simparams):
    """Solve one generated MiniZinc model instance with Gurobi.

    One solver thread is used per process. This is intentional for NCI runs in
    which independent scenarios are distributed across MPI ranks.
    """
    data_file = MINIZINC_DIR / f"hydrogen_plant_data_{simparams['random']}.dzn"
    command = [
        "minizinc",
        "--soln-sep",
        '""',
        "--search-complete-msg",
        '""',
        "--solver",
        "gurobi",
        "--parallel",
        "1",
        "--relGap",
        str(GUROBI_RELATIVE_GAP),
        str(MODEL_FILE),
        str(data_file),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return _parse_minizinc_output(completed.stdout)
    except subprocess.CalledProcessError as exc:
        message = (
            "MiniZinc/Gurobi failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        )
        raise RuntimeError(message) from exc
    finally:
        data_file.unlink(missing_ok=True)


def Optimise(
    load,
    cf,
    simparams,
    PV_location,
    Wind_location,
    Area,
    distancePV,
    distanceWind,
    distanceUser,
    distanceStg,
    random_number,
):
    """Build renewable profiles and solve one hydrogen-hub scenario.

    Parameters retain the names used by the original project code so existing
    scenario wrappers continue to work.
    """
    simparams.update(CF=cf)
    storage_type = simparams["storage_type"]

    wind_profiles = []
    for location in Wind_location:
        WindSource_windlab(location, random_number)
        profile = np.asarray(wind_gen(location, random_number), dtype=float)
        wind_profiles.append(np.trunc(100.0 * profile) / 100.0)
    wind_ref_pout = np.vstack(wind_profiles)

    pv_profiles = []
    for location in PV_location:
        SolarResource(location, random_number)
        profile = np.asarray(
            pv_gen(PV_REFERENCE_CAPACITY_KW, random_number), dtype=float
        )
        pv_profiles.append(np.trunc(100.0 * profile) / 100.0)
    pv_ref_pout = np.vstack(pv_profiles)

    initial_ug_capa = 0.0 if storage_type == "Pipeline" else 110.0

    simparams.update(
        DT=1.0,  # hourly time step
        PV_REF=PV_REFERENCE_CAPACITY_KW,
        WIND_REF=WIND_REFERENCE_CAPACITY_KW,
        C_UG_STORAGE=Cost_hs(initial_ug_capa, storage_type)
        * simparams["C_stg_ratio"],
        LOAD=[float(load)] * pv_ref_pout.shape[1],
        CF=cf,
        PV_REF_POUT=pv_ref_pout,
        WIND_REF_POUT=wind_ref_pout,
        Area=Area,
        distancePV=distancePV,
        distanceWind=distanceWind,
        distanceUser=distanceUser,
        distanceStg=distanceStg,
        random=random_number,
    )

    make_dzn_file(**simparams)
    results = Minizinc(simparams)

    # Underground-storage unit cost depends on cavern size. The original
    # workflow performs one refinement if the first solution differs by >5%
    # from the initial size used for the cost estimate.
    if storage_type == "Depleted gas":
        initial_ug_capa = results["ug_storage_capa"][0] / 1e3

    if simparams["UG_STORAGE_CAPA_MAX"] > 0:
        new_ug_capa = results["ug_storage_capa"][0] / 1e3
        mean_capa = np.mean([new_ug_capa, initial_ug_capa])
        if new_ug_capa > 0 and mean_capa > 0:
            relative_change = abs(new_ug_capa - initial_ug_capa) / mean_capa
            if relative_change > 0.05:
                simparams["C_UG_STORAGE"] = (
                    Cost_hs(new_ug_capa, storage_type)
                    * simparams["C_stg_ratio"]
                )
                make_dzn_file(**simparams)
                results = Minizinc(simparams)

    results.update(CF=simparams["CF"], C_UG_STORAGE=simparams["C_UG_STORAGE"])
    return results, simparams


def Cost_hs(size, storage_type):
    """Return hydrogen-storage unit CAPEX in USD/kg-H2.

    The correlations are retained from the original project implementation.
    ``size`` is expressed in the units used by those correlations (the caller
    passes optimisation storage capacity divided by 1e3).
    """
    size = float(size)

    if storage_type in {"Salt Cavern", "Lined Rock"}:
        if size <= 0:
            raise ValueError(
                f"{storage_type} cost correlation requires a positive size."
            )
        x = np.log10(size)
        if size <= 100:
            return 10 ** (-0.0285 * x + 2.7853)
        if storage_type == "Salt Cavern":
            return 17.66 if size > 8000 else 10 ** (
                0.212669 * x**2 - 1.638654 * x + 4.403100
            )
        return 41.48 if size > 4000 else 10 ** (
            0.217956 * x**2 - 1.575209 * x + 4.463930
        )

    if storage_type == "Depleted gas":
        return 2.72 * 0.746
    if storage_type == "Pipeline":
        return 516.0

    raise ValueError(f"Unsupported storage type: {storage_type}")
