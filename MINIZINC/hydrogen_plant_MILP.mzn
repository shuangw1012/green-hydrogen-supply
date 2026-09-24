include "piecewise_linear.mzn";

% =============================================================================
% Renewable hydrogen hub optimisation model
% =============================================================================
% The model co-optimises PV, wind, electrolyser, battery and hydrogen-storage
% capacities together with hourly operation and a discrete electrolyser site.
%
% The formulation is MILP. PV and wind capacities are continuous variables;
% zero capacity means that no plant is built in a candidate cell. Products
% between the binary electrolyser-location variable and continuous
% transmission/pipeline cost variables are represented by exact
% binary-continuous linearisation constraints.
% =============================================================================

% --- Sets and dimensions ------------------------------------------------------
int: N;                       % number of hourly intervals
int: n_PV;                    % number of PV candidate cells
int: n_wind;                  % number of wind candidate cells
int: n_project;               % project lifetime (years; retained as input)

set of int: tint = 1..N;
set of int: tsamp = 1..N+1;
set of int: N_PV = 1..n_PV;
set of int: N_wind = 1..n_wind;
set of int: N_el = 1..n_PV+1; % candidate cells plus the end-user location
set of int: N_cap = 1..5;

% The present spatial dataset uses the same candidate grid for PV and wind.
constraint n_wind = n_PV;

% --- Technical and economic parameters ---------------------------------------
float: DT;                    % interval duration (h)
float: EL_ETA;                % electrolyser efficiency
float: BAT_ETA_in;            % battery charging efficiency
float: BAT_ETA_out;           % battery discharging efficiency

float: C_PV;                  % PV CAPEX (USD/kW)
float: C_WIND;                % wind CAPEX (USD/kW)
float: C_EL;                  % electrolyser CAPEX (USD/kW)
float: C_UG_STORAGE;          % underground H2 storage CAPEX (USD/kg-H2)
float: UG_STORAGE_CAPA_MAX;   % maximum underground H2 storage (kg-H2)
float: C_PIPE_STORAGE;        % line-packing storage CAPEX (USD/kg-H2)
float: PIPE_STORAGE_CAPA_MIN; % minimum line-packing storage (kg-H2)
float: C_BAT_ENERGY;          % battery energy CAPEX (USD/kWh)
float: C_BAT_POWER;           % battery power CAPEX (USD/kW)

float: OM_EL;                 % electrolyser fixed O&M (USD/kW-year)
float: OM_PV;                 % PV fixed O&M (USD/kW-year)
float: OM_WIND;               % wind fixed O&M (USD/kW-year)
float: OM_UG;                 % underground storage fixed O&M (USD/kg-year)

float: DIS_RATE;              % discount rate (retained for data transparency)
float: crf;                   % capital recovery factor
float: H_total;               % annual renewable H2 supplied (kg/year)
float: RES_H_CAPA;            % annual virtual H2 reserve allowance (kg-H2)

float: PV_REF;                % reference PV capacity (kW)
float: WIND_REF;              % reference wind capacity (kW)

array[tint] of float: LOAD;                       % H2 demand (kg/s)
array[N_PV, tint] of float: PV_REF_POUT;          % reference PV output (kW)
array[N_wind, tint] of float: WIND_REF_POUT;      % reference wind output (kW)
array[N_PV] of float: Area;                       % developable area (km2)

array[N_PV, N_el] of float: distancePV;           % PV-to-electrolyser distance (km)
array[N_wind, N_el] of float: distanceWind;       % wind-to-electrolyser distance (km)
array[N_el] of float: distanceUser;               % electrolyser-to-user distance (km)
array[N_el] of float: distanceStg;                % electrolyser-to-storage distance (km)

% Piecewise-linear infrastructure cost curves.
array[N_cap] of float: capacityLevels;
array[N_cap] of float: TranscapexValues;
array[N_cap] of float: PipecapexValues;
float: TransopexRatio;
float: PipeopexRatio;

% Upper bounds used in the exact binary-continuous linearisation. The supplied
% PWL cost curves are non-negative and bounded by their maximum knot values for
% the capacity range used in this case study.
float: M_TRANS_CAPEX = max(TranscapexValues);
float: M_PIPE_CAPEX = max(PipecapexValues);

% --- Capacity and cost decision variables ------------------------------------
var float: pv_max;
var float: wind_max;
var float: el_max;
var float: ug_storage_capa;
var float: pipe_storage_capa;
var float: bat_e_capa;
var float: bat_p_max;

array[N_PV] of var float: pv_max_array;
array[N_wind] of var float: wind_max_array;
array[N_el] of var bool: electrolyserAt;

var float: C_trans;
var float: C_pipe;
var float: capex;
var float: lcoh;
var float: OF;

var float: FOM_PV;
var float: FOM_WIND;
var float: FOM_EL;
var float: FOM_UG;
var float: TransopexValues;
var float: PipeopexValues;

array[N_PV] of var float: newPVcapexTrans;
array[N_wind] of var float: newWindcapexTrans;
array[N_el] of var float: newElcapexPipe;
var float: newUsercapexPipe;

% Auxiliary variables for exact location-dependent cost linearisation.
array[N_PV, N_el] of var float: transCostPV;
array[N_wind, N_el] of var float: transCostWind;
array[N_el] of var float: pipeCostEl;
array[N_el] of var float: pipeCostUser;

% --- Hourly operational variables --------------------------------------------
array[tint] of var float: pv_pout;
array[tint] of var float: wind_pout;
array[tint] of var float: curtail_p;
array[tint] of var float: el_pin_pvwind;
array[tint] of var float: el_pin;
array[tint] of var float: bat_pin;
array[tint] of var float: bat_pout;

array[tint] of var float: comp1_hflow;
array[tint] of var float: comp1_pin;
array[tint] of var float: comp2_hflow;
array[tint] of var float: comp2_pin;
array[tint] of var float: pipe_storage_hout;
array[tint] of var float: ug_storage_hout;
array[tint] of var float: res_hout;

array[tsamp] of var float: ug_storage_level;
array[tsamp] of var float: pipe_storage_level;
array[tsamp] of var float: res_h_level;
array[tsamp] of var float: bat_e;

% =============================================================================
% Objective and cost accounting
% =============================================================================
OF =
    (C_PV + OM_PV / crf) * pv_max
    + (C_WIND + OM_WIND / crf) * wind_max
    + (C_EL + OM_EL / crf) * el_max
    + (C_UG_STORAGE + OM_UG / crf) * ug_storage_capa
    + C_PIPE_STORAGE * pipe_storage_capa
    + C_BAT_ENERGY * bat_e_capa
    + C_BAT_POWER * bat_p_max
    + C_trans + TransopexValues / crf
    + C_pipe + PipeopexValues / crf;

constraint capex =
    C_PV * pv_max
    + C_WIND * wind_max
    + C_EL * el_max
    + C_UG_STORAGE * ug_storage_capa
    + C_PIPE_STORAGE * pipe_storage_capa
    + C_BAT_ENERGY * bat_e_capa
    + C_BAT_POWER * bat_p_max
    + C_trans + C_pipe;

constraint FOM_PV = pv_max * OM_PV;
constraint FOM_WIND = wind_max * OM_WIND;
constraint FOM_EL = el_max * OM_EL;
constraint FOM_UG = ug_storage_capa * OM_UG;
constraint TransopexValues = TransopexRatio * C_trans;
constraint PipeopexValues = PipeopexRatio * C_pipe;

constraint lcoh =
    (crf * capex + FOM_PV + FOM_WIND + FOM_EL + FOM_UG
     + TransopexValues + PipeopexValues) / H_total;

% =============================================================================
% Capacity and spatial constraints
% =============================================================================
constraint pv_max = sum(j in N_PV)(pv_max_array[j]);
constraint wind_max = sum(j in N_wind)(wind_max_array[j]);

constraint pv_max >= 0;
constraint wind_max >= 0;
constraint el_max >= 0;
constraint el_max / 3600 * EL_ETA / 39.4 >= LOAD[1];
constraint el_max <= pv_max + wind_max;

constraint ug_storage_capa >= 0;
constraint ug_storage_capa <= UG_STORAGE_CAPA_MAX;
constraint pipe_storage_capa >= PIPE_STORAGE_CAPA_MIN;
constraint bat_e_capa >= 0;

% Renewable capacities are continuous from zero to their land-based maxima.
% A zero capacity naturally represents a candidate cell that is not developed.
constraint forall(j in N_PV)(
    pv_max_array[j] >= 0 /\
    pv_max_array[j] <= Area[j] * 41.8 * 1e3
);
constraint forall(j in N_wind)(
    wind_max_array[j] >= 0 /\
    wind_max_array[j] <= Area[j] * 5.2 * 1e3
);

% PV and wind share the same developable area in each candidate grid cell.
constraint forall(j in N_PV)(
    wind_max_array[j] / 1e3 / 5.2
    + pv_max_array[j] / 1e3 / 41.8
    <= Area[j]
);

% Exactly one electrolyser location is selected.
constraint sum(k in N_el)(bool2int(electrolyserAt[k])) = 1;

% =============================================================================
% Transmission and pipeline costs: exact binary-continuous linearisation
% =============================================================================
constraint forall(j in N_PV)(
    newPVcapexTrans[j] =
        piecewise_linear(pv_max_array[j], capacityLevels, TranscapexValues) /\
    newPVcapexTrans[j] >= 0
);

constraint forall(j in N_wind)(
    newWindcapexTrans[j] =
        piecewise_linear(wind_max_array[j], capacityLevels, TranscapexValues) /\
    newWindcapexTrans[j] >= 0
);

% transCostPV[j,k] = newPVcapexTrans[j] * electrolyserAt[k]
constraint forall(j in N_PV, k in N_el)(
    transCostPV[j,k] >= 0 /\
    transCostPV[j,k] <= newPVcapexTrans[j] /\
    transCostPV[j,k] <= M_TRANS_CAPEX * bool2int(electrolyserAt[k]) /\
    transCostPV[j,k] >= newPVcapexTrans[j]
        - M_TRANS_CAPEX * (1 - bool2int(electrolyserAt[k]))
);

% transCostWind[j,k] = newWindcapexTrans[j] * electrolyserAt[k]
constraint forall(j in N_wind, k in N_el)(
    transCostWind[j,k] >= 0 /\
    transCostWind[j,k] <= newWindcapexTrans[j] /\
    transCostWind[j,k] <= M_TRANS_CAPEX * bool2int(electrolyserAt[k]) /\
    transCostWind[j,k] >= newWindcapexTrans[j]
        - M_TRANS_CAPEX * (1 - bool2int(electrolyserAt[k]))
);

constraint C_trans =
    sum(j in N_PV, k in N_el)(transCostPV[j,k] * distancePV[j,k])
    + sum(j in N_wind, k in N_el)(transCostWind[j,k] * distanceWind[j,k]);

constraint forall(k in N_el)(
    newElcapexPipe[k] =
        piecewise_linear(el_max * EL_ETA, capacityLevels, PipecapexValues) /\
    newElcapexPipe[k] >= 0
);

% User-pipeline capacity is based on the fixed hydrogen demand.
constraint newUsercapexPipe =
    piecewise_linear(LOAD[1] * 39.4 * 3600, capacityLevels, PipecapexValues);
constraint newUsercapexPipe >= 0;

% pipeCostEl[k] = newElcapexPipe[k] * electrolyserAt[k]
constraint forall(k in N_el)(
    pipeCostEl[k] >= 0 /\
    pipeCostEl[k] <= newElcapexPipe[k] /\
    pipeCostEl[k] <= M_PIPE_CAPEX * bool2int(electrolyserAt[k]) /\
    pipeCostEl[k] >= newElcapexPipe[k]
        - M_PIPE_CAPEX * (1 - bool2int(electrolyserAt[k]))
);

% pipeCostUser[k] = newUsercapexPipe * electrolyserAt[k]
constraint forall(k in N_el)(
    pipeCostUser[k] >= 0 /\
    pipeCostUser[k] <= newUsercapexPipe /\
    pipeCostUser[k] <= M_PIPE_CAPEX * bool2int(electrolyserAt[k]) /\
    pipeCostUser[k] >= newUsercapexPipe
        - M_PIPE_CAPEX * (1 - bool2int(electrolyserAt[k]))
);

constraint C_pipe =
    sum(k in N_el)(pipeCostEl[k] * distanceStg[k])
    + sum(k in N_el)(pipeCostUser[k] * distanceUser[k]);

% =============================================================================
% Hourly electricity balance and battery operation
% =============================================================================
constraint forall(i in tint)(
    pv_pout[i] =
        sum(j in N_PV)(pv_max_array[j] / PV_REF * PV_REF_POUT[j,i])
);
constraint forall(i in tint)(
    wind_pout[i] =
        sum(j in N_wind)(wind_max_array[j] / WIND_REF * WIND_REF_POUT[j,i])
);

constraint forall(i in tint)(
    pv_pout[i] + wind_pout[i] - curtail_p[i] - el_pin_pvwind[i] - bat_pin[i] = 0
);
constraint forall(i in tint)(curtail_p[i] >= 0);

constraint forall(i in tint)(
    el_pin[i] = el_pin_pvwind[i] + bat_pout[i] - comp1_pin[i] - comp2_pin[i]
);
constraint forall(i in tint)(el_pin[i] >= 0);
constraint forall(i in tint)(el_pin_pvwind[i] >= 0);
constraint forall(i in tint)(
    el_pin_pvwind[i] + bat_pout[i] - comp1_pin[i] - comp2_pin[i] <= el_max
);

constraint forall(i in tint)(
    bat_pin[i] >= 0 /\ bat_pin[i] <= bat_p_max
);
constraint forall(i in tint)(
    bat_pout[i] >= 0 /\ bat_pout[i] <= bat_p_max
);
constraint forall(i in tint)(
    bat_e[i+1] = bat_e[i]
        + (bat_pin[i] * BAT_ETA_in - bat_pout[i] / BAT_ETA_out) * DT
);
constraint forall(i in tsamp)(bat_e[i] >= 0 /\ bat_e[i] <= bat_e_capa);

% Cyclic boundary conditions retained from the original implementation.
constraint bat_e[1] = bat_e[N];

% =============================================================================
% Hydrogen production, compression, storage and delivery
% =============================================================================
constraint forall(i in tint)(comp1_pin[i] = comp1_hflow[i] * 0.83 * 3600);
constraint forall(i in tint)(comp2_pin[i] = comp2_hflow[i] * 0.41 * 3600);

% Higher heating value of hydrogen = 39.4 kWh/kg.
constraint forall(i in tint)(
    comp1_hflow[i] = el_pin[i] / 3600 * EL_ETA / 39.4
);
constraint forall(i in tint)(
    comp2_hflow[i] >= 0 /\
    comp2_hflow[i] <= el_max / 3600 * EL_ETA / 39.4
);

constraint forall(i in tsamp)(
    ug_storage_level[i] >= 0 /\ ug_storage_level[i] <= ug_storage_capa
);
constraint forall(i in tsamp)(
    pipe_storage_level[i] >= 0 /\ pipe_storage_level[i] <= pipe_storage_capa
);

constraint forall(i in tint)(
    (comp2_hflow[i] + ug_storage_hout[i]) * DT * 3600 <= ug_storage_capa
);
constraint forall(i in tint)(
    ug_storage_level[i+1] = ug_storage_level[i]
        + (comp2_hflow[i] - ug_storage_hout[i]) * DT * 3600
);
constraint forall(i in tint)(
    pipe_storage_level[i+1] = pipe_storage_level[i]
        + (comp1_hflow[i] - pipe_storage_hout[i] - comp2_hflow[i]) * DT * 3600
);

constraint forall(i in tint)(
    pipe_storage_hout[i] + ug_storage_hout[i] + res_hout[i] = LOAD[i]
);
constraint forall(i in tint)(pipe_storage_hout[i] >= 0);
constraint forall(i in tint)(ug_storage_hout[i] >= 0);

% Cyclic boundary conditions retained from the original implementation.
constraint ug_storage_level[1] = ug_storage_level[N];
constraint pipe_storage_level[1] = pipe_storage_level[N];

% =============================================================================
% Virtual hydrogen reserve / annual supply-reliability allowance
% =============================================================================
% The reserve starts with RES_H_CAPA and can only be depleted. Because its
% state is tracked through the full year and constrained to remain non-negative,
% cumulative virtual hydrogen use cannot exceed the annual reserve allowance.
constraint res_h_level[1] = RES_H_CAPA;
constraint forall(i in tsamp)(
    res_h_level[i] >= 0 /\ res_h_level[i] <= RES_H_CAPA
);
constraint forall(i in tint)(
    res_h_level[i+1] = res_h_level[i] - res_hout[i] * DT * 3600
);
constraint forall(i in tint)(
    res_hout[i] >= 0 /\ res_hout[i] <= LOAD[i]
);

% =============================================================================
% Solve and output
% =============================================================================
solve minimize OF;

% Output names are intentionally retained because the Python wrapper parses
% these exact keys.
output ["!"] ++
    ["N="] ++ [show(N)] ++ [";"] ++
    ["CAPEX="] ++ [show(capex)] ++ [";"] ++
    ["pv_max="] ++ [show(pv_max)] ++ [";"] ++
    ["wind_max="] ++ [show(wind_max)] ++ [";"] ++
    ["pv_max_array="] ++ [show(pv_max_array)] ++ [";"] ++
    ["wind_max_array="] ++ [show(wind_max_array)] ++ [";"] ++
    ["el_max="] ++ [show(el_max)] ++ [";"] ++
    ["ug_storage_capa="] ++ [show(ug_storage_capa)] ++ [";"] ++
    ["pipe_storage_capa="] ++ [show(pipe_storage_capa)] ++ [";"] ++
    ["bat_e_capa="] ++ [show(bat_e_capa)] ++ [";"] ++
    ["bat_p_max="] ++ [show(bat_p_max)] ++ [";"] ++
    ["pv_pout="] ++ [show(pv_pout)] ++ [";"] ++
    ["wind_pout="] ++ [show(wind_pout)] ++ [";"] ++
    ["curtail_p="] ++ [show(curtail_p)] ++ [";"] ++
    ["bat_pin="] ++ [show(bat_pin)] ++ [";"] ++
    ["bat_pout="] ++ [show(bat_pout)] ++ [";"] ++
    ["el_pin="] ++ [show(el_pin)] ++ [";"] ++
    ["comp1_hflow="] ++ [show(comp1_hflow)] ++ [";"] ++
    ["comp1_pin="] ++ [show(comp1_pin)] ++ [";"] ++
    ["comp2_hflow="] ++ [show(comp2_hflow)] ++ [";"] ++
    ["comp2_pin="] ++ [show(comp2_pin)] ++ [";"] ++
    ["res_hout="] ++ [show(res_hout)] ++ [";"] ++
    ["el_pin_pvwind="] ++ [show(el_pin_pvwind)] ++ [";"] ++
    ["pipe_storage_hout="] ++ [show(pipe_storage_hout)] ++ [";"] ++
    ["ug_storage_hout="] ++ [show(ug_storage_hout)] ++ [";"] ++
    ["ug_storage_level="] ++ [show(ug_storage_level)] ++ [";"] ++
    ["pipe_storage_level="] ++ [show(pipe_storage_level)] ++ [";"] ++
    ["reserve_h_level="] ++ [show(res_h_level)] ++ [";"] ++
    ["bat_e="] ++ [show(bat_e)] ++ [";"] ++
    ["LOAD="] ++ [show(LOAD)] ++ [";"] ++
    ["C_trans="] ++ [show(C_trans)] ++ [";"] ++
    ["C_pipe="] ++ [show(C_pipe)] ++ [";"] ++
    ["lcoh="] ++ [show(lcoh)] ++ [";"] ++
    ["FOM_PV="] ++ [show(FOM_PV)] ++ [";"] ++
    ["FOM_WIND="] ++ [show(FOM_WIND)] ++ [";"] ++
    ["FOM_EL="] ++ [show(FOM_EL)] ++ [";"] ++
    ["FOM_UG="] ++ [show(FOM_UG)] ++ [";"] ++
    ["FOM_TRANS="] ++ [show(TransopexValues)] ++ [";"] ++
    ["FOM_PIPE="] ++ [show(PipeopexValues)] ++ [";"] ++
    ["H_total="] ++ [show(H_total)] ++ [";"] ++
    ["El_location="] ++
        [show(k) | k in N_el where fix(electrolyserAt[k]) = true] ++ [";"] ++
    ["newWindcapexTrans="] ++ [show(newWindcapexTrans)] ++ [";"] ++
    ["!"];
