"""Renewable generation models and weather-file preparation.

This module prepares the temporary resource files required by NREL PySAM and
returns hourly reference generation profiles for the optimisation model.
The published Tasmania case uses PVWatts v8 for PV and Windpower for wind.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as PVWatts
import PySAM.Windpower as Windpower

from projdirs import DATA_DIR

SOLAR_DIR = DATA_DIR / "SAM_INPUTS" / "SOLAR"
WIND_DIR = DATA_DIR / "SAM_INPUTS" / "WIND"
WEATHER_DIR = DATA_DIR / "SAM_INPUTS" / "WEATHER_DATA"
WIND_RESULTS_DIR = WIND_DIR / "SAM_results"

PV_CONFIG = SOLAR_DIR / "pvfarm_pvwattsv8.json"
WIND_CONFIG = WIND_DIR / "windfarm_windpower.json"


def pv_gen(capacity: float, run_id: str) -> list[float]:
    """Run PySAM PVWatts for the prepared solar resource file.

    Parameters
    ----------
    capacity
        Reference PV capacity in kW.
    run_id
        Unique identifier used for temporary files. This prevents collisions
        when several scenarios are run concurrently on NCI.

    Returns
    -------
    list[float]
        Hourly PV generation in kW for all years contained in the source file.
    """
    source_file = SOLAR_DIR / f"SolarSource_{run_id}.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Solar resource file not found: {source_file}")

    source = pd.read_csv(source_file, low_memory=False)
    n_years = int((len(source) - 2) / 8760)
    if n_years < 1:
        raise ValueError(f"Unexpected solar resource length in {source_file}")

    pv = PVWatts.new()
    output_all: list[float] = []
    single_year_file = SOLAR_DIR / f"SolarSource_{run_id}_single_year.csv"

    try:
        for year_index in range(n_years):
            start = 2 + 8760 * year_index
            stop = start + 8760
            one_year = pd.concat([source.iloc[:2], source.iloc[start:stop]])
            one_year.to_csv(single_year_file, index=False, lineterminator="\n")

            with PV_CONFIG.open("r", encoding="utf-8") as stream:
                config = json.load(stream)

            # The resource path is scenario-specific and therefore overrides
            # the path stored in the exported PySAM configuration file.
            config["solar_resource_file"] = str(single_year_file)
            config["system_capacity"] = float(capacity)

            for key, value in config.items():
                if key != "number_inputs":
                    pv.value(key, value)

            pv.execute()
            output_all.extend(np.asarray(pv.Outputs.gen, dtype=float).tolist())
    finally:
        single_year_file.unlink(missing_ok=True)

    return output_all


def wind_gen(loc: str, run_id: str, hub_height: float = 150.0) -> list[float]:
    """Return hourly wind generation for a candidate cell.

    Cached PySAM results are used when available. Otherwise, the prepared SRW
    file for ``run_id`` is passed to PySAM Windpower and the result is cached.

    Parameters
    ----------
    loc
        Location identifier, including the weather-data year suffix.
    run_id
        Unique identifier for temporary weather files.
    hub_height
        Wind turbine hub height in metres.
    """
    WIND_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = WIND_RESULTS_DIR / f"{loc}.csv"

    if cache_file.exists():
        output = np.loadtxt(cache_file, delimiter=",")
        return np.asarray(output, dtype=float).tolist()

    source_file = WIND_DIR / f"WindSource_{run_id}.srw"
    if not source_file.exists():
        raise FileNotFoundError(f"Wind resource file not found: {source_file}")

    wind = Windpower.new()
    with WIND_CONFIG.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    config["wind_resource_filename"] = str(source_file)
    for key, value in config.items():
        if key != "number_inputs":
            wind.value(key, value)

    wind.Turbine.wind_turbine_hub_ht = float(hub_height)
    wind.execute()
    output = np.asarray(wind.Outputs.gen, dtype=float)

    # Write through a temporary file before replacing the shared cache. This
    # avoids leaving a partial cache file if a process is interrupted.
    temp_cache = WIND_RESULTS_DIR / f".{loc}.{run_id}.tmp"
    np.savetxt(temp_cache, output, delimiter=",")
    os.replace(temp_cache, cache_file)
    return output.tolist()


def SolarResource(location: str, run_id: str) -> None:
    """Prepare a candidate-cell weather file for PySAM PVWatts."""
    source = WEATHER_DIR / f"weather_data_{location}.csv"
    target = SOLAR_DIR / f"SolarSource_{run_id}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Weather file not found: {source}")

    # Retain the original project behaviour: parse and re-write the SAM CSV
    # before passing it to PVWatts.
    data = pd.read_csv(source, low_memory=False)
    data.to_csv(target, index=False, lineterminator="\n")


def WindSource_windlab(location: str, run_id: str) -> None:
    """Create a PySAM SRW file from the Windlab-format weather data.

    The supplied wind resource is at 150 m. A 10 m layer is added using a
    logarithmic wind profile because the PySAM Windpower resource format used
    here contains both heights.
    """
    source = WEATHER_DIR / f"weather_data_{location}.csv"
    target = WIND_DIR / f"WindSource_{run_id}.srw"
    if not source.exists():
        raise FileNotFoundError(f"Weather file not found: {source}")

    metadata = pd.read_csv(source, nrows=1, low_memory=False)
    latitude = metadata.loc[0, "lat"]
    longitude = metadata.loc[0, "lon"]

    weather = pd.read_csv(source, skiprows=2)
    data_150 = weather.iloc[:, [5, 14, 15, 16]].copy()
    data_150["Pressure"] = data_150["Pressure"] / 1013.25
    data_150 = data_150.rename(
        columns={
            "Temperature": "T",
            "Wind Speed": "S",
            "Wind Direction": "D",
            "Pressure": "P",
        }
    )

    heading_150 = pd.DataFrame(
        {
            "T": ["Temperature", "C", 150],
            "S": ["Speed", "m/s", 150],
            "D": ["Direction", "degrees", 150],
            "P": ["Pressure", "atm", 150],
        }
    )
    data_150 = pd.concat([heading_150, data_150], ignore_index=True)

    data_10 = data_150.copy()
    data_10.iloc[2, :] = 10
    data_10_body = data_10.iloc[3:].copy()
    data_10_body["S"] = speed(10.0, 150.0, data_10_body["S"])
    data_10 = pd.concat([data_10.iloc[:3], data_10_body], ignore_index=True)

    data = pd.concat([data_150, data_10], axis=1)

    # PySAM SRW metadata rows contain one entry per resource column.
    n_columns = data.shape[1]
    data.loc[-1] = [f"Latitude:{latitude}"] * n_columns
    data.index = data.index + 1
    data.sort_index(inplace=True)
    data.loc[-1] = [f"Longitude:{longitude}"] * n_columns
    data.index = data.index + 1
    data.sort_index(inplace=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, header=False, index=False, lineterminator="\n")


def speed(z: float, z_anem: float, u_anem):
    """Scale wind speed between heights using a logarithmic wind profile."""
    roughness_length = 0.01
    return u_anem * np.log(z / roughness_length) / np.log(
        z_anem / roughness_length
    )
