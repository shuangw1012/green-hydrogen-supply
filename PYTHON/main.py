#!/usr/bin/env python3
"""Run renewable-hydrogen hub optimisation scenarios.

Each row in ``input.txt`` defines one scenario. When launched under MPI, rows
are distributed across ranks so independent scenarios can be solved in
parallel. With 17 rows and 17 MPI ranks, each rank solves one scenario.
"""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from projdirs import DATA_DIR, OUTPUT_DIR, PYTHON_DIR
from PACKAGE.optimisation import Optimise

SITE_FILE = PYTHON_DIR / "Input_Tas.txt"
SCENARIO_FILE = PYTHON_DIR / "input.txt"

KM_PER_DEGREE = 111.32
ROUTE_DETOUR_FACTOR = 1.2
PROJECT_LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.06
WRITE_TIME_SERIES = True


def _capital_recovery_factor(rate: float, years: int) -> float:
    """Return the standard capital recovery factor."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def _euclidean_distance_km(lat_a, lon_a, lat_b, lon_b):
    """Approximate point-to-point distance using the model's original method."""
    return (
        np.sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2)
        * KM_PER_DEGREE
        * ROUTE_DETOUR_FACTOR
    )


def _prepare_spatial_inputs(site_data: pd.DataFrame):
    """Build candidate locations, available area, and distance matrices."""
    # The final two rows are the end user and hydrogen-storage location.
    candidate_rows = site_data.iloc[:-2]
    electrolyser_rows = site_data.iloc[:-1]  # candidates + end-user location

    pv_locations = candidate_rows["#Name"].to_numpy()
    wind_locations = pv_locations.copy()
    electrolyser_locations = electrolyser_rows["#Name"].to_numpy()
    area = candidate_rows["Area"].astype(float).tolist()

    candidate_lat = candidate_rows["Lat"].to_numpy(dtype=float)[:, np.newaxis]
    candidate_lon = candidate_rows["Long"].to_numpy(dtype=float)[:, np.newaxis]
    el_lat = electrolyser_rows["Lat"].to_numpy(dtype=float)
    el_lon = electrolyser_rows["Long"].to_numpy(dtype=float)

    distance_pv = _euclidean_distance_km(
        candidate_lat, candidate_lon, el_lat, el_lon
    )
    distance_wind = distance_pv.copy()

    user_row = site_data.loc[site_data["#Name"] == "User"].iloc[0]
    storage_row = site_data.loc[site_data["#Name"] == "storage"].iloc[0]

    distance_user = _euclidean_distance_km(
        float(user_row["Lat"]),
        float(user_row["Long"]),
        el_lat,
        el_lon,
    )

    # A storage latitude of zero is the existing input convention for cases
    # without a spatially constrained underground-storage location.
    if float(storage_row["Lat"]) == 0:
        distance_storage = np.zeros_like(el_lat, dtype=float)
    else:
        distance_storage = _euclidean_distance_km(
            float(storage_row["Lat"]),
            float(storage_row["Long"]),
            el_lat,
            el_lon,
        )

    return (
        pv_locations,
        wind_locations,
        electrolyser_locations,
        area,
        distance_pv,
        distance_wind,
        distance_user,
        distance_storage,
    )


def _scenario_parameters(row: pd.Series) -> dict:
    """Create the optimisation-parameter dictionary for one scenario."""
    return {
        "EL_ETA": 0.70,
        "BAT_ETA_in": 0.95,
        "BAT_ETA_out": 0.95,
        "C_PV": float(row["PV_capex"]),
        "C_WIND": float(row["Wind_capex"]),
        "C_EL": float(row["El_capex"]),
        "UG_STORAGE_CAPA_MAX": 1e10,
        "C_PIPE_STORAGE": 516.0,
        "PIPE_STORAGE_CAPA_MIN": 0.0,
        "C_BAT_ENERGY": 196.0 * float(row["Bat_ratio"]),
        "C_BAT_POWER": 405.0 * float(row["Bat_ratio"]),
        "OM_EL": float(row["EL_FOM"]),
        "OM_PV": float(row["PV_FOM"]),
        "OM_WIND": float(row["Wind_FOM"]),
        "OM_UG": 1.03 * float(row["Stg_ratio"]),
        "DIS_RATE": DISCOUNT_RATE,
        "C_stg_ratio": float(row["Stg_ratio"]),
        "C_trans_ratio": float(row["Trans_capex"]),
        "C_pipe_ratio": float(row["Pipe_capex"]),
        "storage_type": row["Stg"],
    }


def _scenario_tag(row: pd.Series) -> str:
    """Return the legacy scenario identifier used in output filenames."""
    fields = [
        "Hub",
        "Stg",
        "load",
        "data_year",
        "CF",
        "cost_year",
        "PV_capex",
        "Wind_capex",
        "El_capex",
        "UG_capex",
        "PV_FOM",
        "Wind_FOM",
        "EL_FOM",
        "Stg_ratio",
        "Bat_ratio",
        "Trans_capex",
        "Pipe_capex",
    ]
    return "_".join(str(row[field]) for field in fields)


def _build_summary_row(
    row: pd.Series,
    results: dict,
    simparams: dict,
    electrolyser_locations,
    pv_locations,
    wind_locations,
) -> dict:
    """Collect scalar model outputs and LCOH component contributions."""
    h_total = results["H_total"][0]
    crf = _capital_recovery_factor(DISCOUNT_RATE, PROJECT_LIFETIME_YEARS)

    summary = {
        "Hub": row["Hub"],
        "Stg": row["Stg"],
        "data_year": row["data_year"],
        "CF": row["CF"],
        "cost_year": row["cost_year"],
        "PV_capex": row["PV_capex"],
        "Wind_capex": row["Wind_capex"],
        "El_capex": row["El_capex"],
        "UG_capex": row["UG_capex"],
        "PV_FOM": row["PV_FOM"],
        "Wind_FOM": row["Wind_FOM"],
        "EL_FOM": row["EL_FOM"],
        "Stg_ratio": row["Stg_ratio"],
        "Bat_ratio": row["Bat_ratio"],
        "Trans_capex": row["Trans_capex"],
        "Pipe_capex": row["Pipe_capex"],
        "El": electrolyser_locations[int(results["El_location"][0]) - 1],
        "capex[USD]": results["CAPEX"][0],
        "lcoh[USD/kg]": results["lcoh"][0],
        "FOM_PV[USD]": results["FOM_PV"][0],
        "FOM_WIND[USD]": results["FOM_WIND"][0],
        "FOM_EL[USD]": results["FOM_EL"][0],
        "FOM_UG[USD]": results["FOM_UG"][0],
        "FOM_TRANS[USD]": results["FOM_TRANS"][0],
        "FOM_PIPE[USD]": results["FOM_PIPE"][0],
        "H_total[kg]": h_total,
        "pv_capacity[kW]": results["pv_max"][0],
        "wind_capacity[kW]": results["wind_max"][0],
        "el_capacity[kW]": results["el_max"][0],
        "ug_capcaity[kgH2]": results["ug_storage_capa"][0],
        "pipe_storage_capacity[kgH2]": results["pipe_storage_capa"][0],
        "bat_e_capacity[kWh]": results["bat_e_capa"][0],
        "bat_p_capacity[kW]": results["bat_p_max"][0],
        "pv_cost[USD]": results["pv_max"][0] * simparams["C_PV"],
        "wind_cost[USD]": results["wind_max"][0] * simparams["C_WIND"],
        "el_cost[USD]": results["el_max"][0] * simparams["C_EL"],
        "ug_storage_cost[USD]": results["ug_storage_capa"][0]
        * simparams["C_UG_STORAGE"],
        "pipe_storage_cost[USD]": results["pipe_storage_capa"][0]
        * simparams["C_PIPE_STORAGE"],
        # Reporting-only total battery CAPEX (energy + power components).
        "bat_cost[USD]": results["bat_e_capa"][0] * simparams["C_BAT_ENERGY"]
        + results["bat_p_max"][0] * simparams["C_BAT_POWER"],
        "load[kg/s]": results["LOAD"][0],
        "C_trans[USD]": results["C_trans"][0],
        "C_pipe[USD]": results["C_pipe"][0],
    }

    summary["LCOH-PV"] = (
        crf * results["pv_max"][0] * simparams["C_PV"] + results["FOM_PV"][0]
    ) / h_total
    summary["LCOH-wind"] = (
        crf * results["wind_max"][0] * simparams["C_WIND"]
        + results["FOM_WIND"][0]
    ) / h_total
    summary["LCOH-el"] = (
        crf * results["el_max"][0] * simparams["C_EL"] + results["FOM_EL"][0]
    ) / h_total
    summary["LCOH-UG"] = (
        crf * results["ug_storage_capa"][0] * simparams["C_UG_STORAGE"]
        + results["FOM_UG"][0]
    ) / h_total
    summary["LCOH-pipe-storage"] = (
        crf * results["pipe_storage_capa"][0] * simparams["C_PIPE_STORAGE"]
    ) / h_total
    summary["LCOH-trans"] = (
        crf * results["C_trans"][0] + results["FOM_TRANS"][0]
    ) / h_total
    summary["LCOH-pipe"] = (
        crf * results["C_pipe"][0] + results["FOM_PIPE"][0]
    ) / h_total

    for index, value in enumerate(results["pv_max_array"]):
        summary[f"pv_capacity_{pv_locations[index]}[kW]"] = value
    for index, value in enumerate(results["wind_max_array"]):
        summary[f"wind_capacity_{wind_locations[index]}[kW]"] = value

    return summary


def _write_time_series(results: dict, output_file: Path) -> None:
    """Write hourly dispatch and storage results for one scenario."""
    data = pd.DataFrame(
        {
            "pipe_storage_level": results["pipe_storage_level"][:-1],
            "ug_storage_level": results["ug_storage_level"][:-1],
            "pv_pout": results["pv_pout"],
            "wind_output": results["wind_pout"],
            "curtail_p": results["curtail_p"],
            "bat_pin": results["bat_pin"],
            "bat_pout": results["bat_pout"],
            "el_pin_pvwind": results["el_pin_pvwind"],
            "el_pin": results["el_pin"],
            "comp1_pin": results["comp1_pin"],
            "comp2_pin": results["comp2_pin"],
            "pipe_storage_hout": results["pipe_storage_hout"],
            "ug_storage_hout": results["ug_storage_hout"],
            "comp1_hflow": results["comp1_hflow"],
            "comp2_hflow": results["comp2_hflow"],
            "res_hout": results["res_hout"],
            "LOAD": results["LOAD"],
        }
    )
    data.to_csv(output_file, index=False)


def _cleanup_temporary_weather_files(run_id: str) -> None:
    """Remove scenario-specific PySAM resource files."""
    (DATA_DIR / "SAM_INPUTS" / "SOLAR" / f"SolarSource_{run_id}.csv").unlink(
        missing_ok=True
    )
    (DATA_DIR / "SAM_INPUTS" / "WIND" / f"WindSource_{run_id}.srw").unlink(
        missing_ok=True
    )


def optimisation(df_slice: pd.DataFrame) -> None:
    """Run every scenario contained in ``df_slice`` and write its outputs."""
    site_data = pd.read_csv(SITE_FILE)
    spatial = _prepare_spatial_inputs(site_data)
    (
        base_pv_locations,
        base_wind_locations,
        electrolyser_locations,
        area,
        distance_pv,
        distance_wind,
        distance_user,
        distance_storage,
    ) = spatial

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df_slice.iterrows():
        simparams = _scenario_parameters(row)
        load = float(row["load"])
        data_year = int(row["data_year"])
        capacity_factor = float(row["CF"])
        run_id = uuid.uuid4().hex

        pv_locations = np.array(
            [f"{name}_{data_year}" for name in base_pv_locations], dtype=str
        )
        wind_locations = np.array(
            [f"{name}_{data_year}" for name in base_wind_locations], dtype=str
        )

        try:
            results, simparams = Optimise(
                load,
                capacity_factor,
                simparams,
                pv_locations,
                wind_locations,
                area,
                distance_pv,
                distance_wind,
                distance_user,
                distance_storage,
                run_id,
            )

            summary = _build_summary_row(
                row,
                results,
                simparams,
                electrolyser_locations,
                pv_locations,
                wind_locations,
            )
            tag = _scenario_tag(row)

            if WRITE_TIME_SERIES:
                _write_time_series(
                    results,
                    OUTPUT_DIR / f"output_2020_{tag}.csv",
                )

            pd.DataFrame([summary]).to_csv(
                OUTPUT_DIR / f"results_2020_{tag}.csv",
                index=False,
            )
        finally:
            _cleanup_temporary_weather_files(run_id)


def _mpi_context():
    """Return ``(comm, rank, size)``; fall back to serial execution."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1

    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def main() -> int:
    """Distribute scenarios across MPI ranks and run them."""
    scenarios = pd.read_csv(SCENARIO_FILE)
    comm, rank, size = _mpi_context()
    failures: list[str] = []

    for case_index in range(rank, len(scenarios), size):
        row = scenarios.iloc[[case_index]]
        print(
            f"[rank {rank}/{size}] starting scenario {case_index + 1}/{len(scenarios)}",
            flush=True,
        )
        try:
            optimisation(row)
            print(
                f"[rank {rank}/{size}] finished scenario {case_index + 1}",
                flush=True,
            )
        except Exception:
            message = (
                f"rank {rank}, scenario {case_index + 1}\n{traceback.format_exc()}"
            )
            failures.append(message)
            print(message, flush=True)

    if comm is not None:
        gathered = comm.gather(failures, root=0)
        if rank == 0:
            all_failures = [item for group in gathered for item in group]
            if all_failures:
                print(f"{len(all_failures)} scenario(s) failed.", flush=True)
                return 1
            print("All scenarios completed successfully.", flush=True)
        return 0

    if failures:
        print(f"{len(failures)} scenario(s) failed.", flush=True)
        return 1
    print("All scenarios completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
