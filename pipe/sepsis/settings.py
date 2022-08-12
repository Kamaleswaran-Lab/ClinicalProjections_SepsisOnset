import pipe.core.core as pc
import pandas as pd
import math
import os.path
from gurobipy import GRB


"""
These are domain-specific settings for use in processing the Physionet dataset
concerning sepsis patients.

See here: https://physionet.org/content/challenge-2019/1.0.0/

QP model overview:
- define two QP models; "physical" representing physically possible values for
  a patient's vitals to take, and "normal" representing values that these vitals
  would be for a normal patient
- the "physical" constraints are used to a) correct the data for anomalies and
  b) flag the user that those anamalies occurred
- the "normal" constraints are used to determine the degree to which the
  patient is abnormal (which may or may not correspond to sepsis likelihood)

Columns overview:
- use all columns in the dataframe except Age, Gender, Unit1, Unit2, 
  HospAdmTime, and ICULOS
- linearly scale all columns since ranges are very different
- log-transform most columns that are bounded between 0 and infinity (eg most
  labs, since these output very large numbers when positive)
- compute a few new columns corresponding to SOFA scores (given the data we
  have, this is not very accurate)
- each column has a set of constraints for the two QP models described above
  which set the lower/upper bound of what is physically possible and/or normal
"""

PHY_MOD_NAME = "phy"
NORM_MOD_NAME = "norm"

# def compute_SOFA_resp(o2Sat, fiO2):
#     fiO2 = fiO2 if fiO2 >= 0.21 else 0.21
#     ratio = o2Sat / fiO2
#     if ratio >= 400:
#         return 0
#     if ratio >= 300:
#         return 1
#     if ratio >= 200:
#         return 2
#     # TODO do they need to be on a ventilator for 3-4 to make sense?
#     if ratio >= 100:
#         return 3
#     return 4


def compute_SOFA_coag(platelets: float) -> float:
    if platelets >= 150:
        return 0
    if platelets >= 100:
        return 1
    if platelets >= 50:
        return 2
    if platelets >= 20:
        return 3
    return 4


def compute_SOFA_liver(bilirubin_total: float) -> float:
    if bilirubin_total <= 1.2:
        return 0
    if bilirubin_total <= 2.0:
        return 1
    if bilirubin_total <= 6.0:
        return 2
    if bilirubin_total <= 12:
        return 3
    return 4


def compute_SOFA_renal(creatinine: float) -> float:
    if creatinine <= 1.2:
        return 0
    if creatinine <= 2.0:
        return 1
    if creatinine <= 3.5:
        return 2
    if creatinine <= 5.0:
        return 3
    return 4


def compute_SOFA(df):
    return df.apply(
        lambda df: max(
            # compute_SOFA_resp(df["FiO2"], df["O2Sat"]),
            compute_SOFA_coag(df["Platelets"]),
            compute_SOFA_liver(df["Bilirubin_total"]),
            compute_SOFA_renal(df["Creatinine"]),
        ),
        axis=1,
    )


def compute_SIRS(df):
    return df.apply(
        lambda df: ((df["Temp"] < 36) | (df["Temp"] > 38))
        + (df["HR"] > 90)
        + ((df["WBC"] < 4) | (df["WBC"] > 12))
        + (df["Resp"] > 20),
        axis=1,
    )


COLUMNS = [
    pc.ColumnSettings(
        "HR",
        impute_type="linear",
        impute_default=75,
        scale_limits=(60, 90),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(30, 200)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(60, 90)),
        ],
    ),
    # this is also called SpO2
    pc.ColumnSettings(
        "O2Sat",
        impute_type="linear",
        impute_default=97.5,
        scale_limits=(95, 100),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(50, 100)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(95, 100)),
        ],
    ),
    pc.ColumnSettings(
        "Temp",
        impute_type="linear",
        impute_default=37,
        scale_limits=(36, 38),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(25, 45), offset_limit=(-2, 2)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(36, 38)),
        ],
    ),
    pc.ColumnSettings(
        "SBP",
        impute_type="linear",
        impute_default=110,
        scale_limits=(90, 130),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(50, 200)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(90, 130)),
        ],
    ),
    pc.ColumnSettings(
        "MAP",
        impute_type="linear",
        impute_default=70,
        scale_limits=(65, 75),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(20, 140)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(65, 75)),
        ],
    ),
    pc.ColumnSettings(
        "DBP",
        impute_type="linear",
        impute_default=70,
        scale_limits=(60, 80),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(20, 150)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(60, 80)),
        ],
    ),
    pc.ColumnSettings(
        "Resp",
        impute_type="linear",
        impute_default=16,
        scale_limits=(10, 24),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(8, 70), offset_limit=(-40, 40)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(10, 24)),
        ],
    ),
    pc.ColumnSettings(
        "EtCO2",
        impute_type="linear",
        impute_default=40,
        scale_limits=(35, 45),
        offset=1,
        constraints=[
            pc.ConstraintSettings(
                PHY_MOD_NAME, limit=(-10, 80), offset_limit=(-30, 30)
            ),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(35, 45)),
        ],
    ),
    pc.ColumnSettings(
        "BaseExcess",
        impute_type="linear",
        impute_default=0,
        scale_limits=(-2, 2),
        offset=1,
        constraints=[
            pc.ConstraintSettings(
                PHY_MOD_NAME, limit=(-40, 20), offset_limit=(-10, 10)
            ),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(-2, 2)),
        ],
    ),
    pc.ColumnSettings(
        "HCO3",
        impute_type="linear",
        impute_default=24.5,
        scale_limits=(22, 27),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 45)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(22, 27)),
        ],
    ),
    pc.ColumnSettings(
        "FiO2",
        impute_type="stepwise",
        impute_default=0,
        scale_limits=(0, 0.2),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 1.0)),
            # this is a dirty hack, if some is "normal" they should have a
            # value of 0, and if someone is on a ventilator they should have
            # a value of 0.21 or above. Anything between 0 and 0.21 makes no
            # sense. However, we want 0.21+ to contribute to distance from
            # normal, so we need to set the upper limit to be less than 0.21
            # since 0.21 is actually an abnormal value.
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0, 0.2)),
        ],
    ),
    pc.ColumnSettings(
        "pH",
        impute_type="linear",
        impute_default=7.4,
        scale_limits=(7.35, 7.45),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(6.5, 7.7)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(7.35, 7.45)),
        ],
    ),
    pc.ColumnSettings(
        "PaCO2",
        impute_type="linear",
        impute_default=40,
        scale_limits=(35, 45),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(16, 120)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(35, 45)),
        ],
    ),
    # also known as PaO2
    pc.ColumnSettings(
        "SaO2",
        impute_type="linear",
        impute_default=97.5,
        scale_limits=(95, 100),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(50, 100)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(95, 100)),
        ],
    ),
    pc.ColumnSettings(
        "Calcium",
        impute_type="linear",
        impute_default=9.5,
        scale_limits=(8.5, 10.5),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(4, 20)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(8.5, 10.5)),
        ],
    ),
    pc.ColumnSettings(
        "Chloride",
        impute_type="linear",
        impute_default=101,
        scale_limits=(96, 106),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(50, 150)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(96, 106)),
        ],
    ),
    pc.ColumnSettings(
        "Creatinine",
        impute_type="linear",
        impute_default=0.9,
        scale_limits=(0.5, 1.3),
        log_transform=True,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 20)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0.5, 1.3)),
        ],
    ),
    pc.ColumnSettings(
        "Bilirubin_direct",
        impute_type="linear",
        impute_default=0.2,
        scale_limits=(0, 0.4),
        log_transform=True,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 50)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0, 0.4)),
        ],
    ),
    pc.ColumnSettings(
        "Glucose",
        impute_type="linear",
        impute_default=130,
        scale_limits=(60, 200),
        log_transform=True,
        offset=1,
        # TODO does this offset limit still make sense in the log-transformed space?
        constraints=[
            pc.ConstraintSettings(
                PHY_MOD_NAME, limit=(10, 3000), offset_limit=(-300, 300)
            ),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(60, 200)),
        ],
    ),
    pc.ColumnSettings(
        "Lactate",
        impute_type="linear",
        impute_default=0.75,
        scale_limits=(0.5, 1.0),
        log_transform=True,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 30)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0.5, 1.0)),
        ],
    ),
    pc.ColumnSettings(
        "Magnesium",
        impute_type="linear",
        impute_default=2.0,
        scale_limits=(1.5, 2.5),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 10)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(1.5, 2.5)),
        ],
    ),
    pc.ColumnSettings(
        "Phosphate",
        impute_type="linear",
        impute_default=3.0,
        scale_limits=(2.5, 4.5),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 12)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(2.5, 4.5)),
        ],
    ),
    pc.ColumnSettings(
        "Potassium",
        impute_type="linear",
        impute_default=4.0,
        scale_limits=(3.5, 4.5),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 12)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(3.5, 4.5)),
        ],
    ),
    pc.ColumnSettings(
        "Bilirubin_total",
        impute_type="linear",
        impute_default=0.7,
        scale_limits=(0.2, 1.2),
        log_transform=True,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 80)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0.2, 1.2)),
        ],
    ),
    pc.ColumnSettings(
        "TroponinI",
        impute_type="linear",
        impute_default=0.15,
        scale_limits=(0, 0.3),
        log_transform=True,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 80)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0, 0.3)),
        ],
    ),
    pc.ColumnSettings(
        "Hct",
        impute_type="linear",
        impute_default=40,
        scale_limits=(35, 45),
        offset=1,
        constraints=[
            # this might have 60 for upper bound (apache)
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(10, 50)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(35, 45)),
        ],
    ),
    pc.ColumnSettings(
        "Hgb",
        impute_type="linear",
        impute_default=14.5,
        scale_limits=(12, 17),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 17)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(12, 17)),
        ],
    ),
    pc.ColumnSettings(
        "PTT",
        impute_type="linear",
        impute_default=65,
        scale_limits=(60, 70),
        log_transform=False,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 200)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(60, 70)),
        ],
    ),
    pc.ColumnSettings(
        "WBC",
        impute_type="linear",
        impute_default=7.5,
        scale_limits=(4, 11),
        log_transform=True,
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 200)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(4, 11)),
        ],
    ),
    pc.ColumnSettings(
        "Fibrinogen",
        impute_type="linear",
        impute_default=300,
        scale_limits=(200, 400),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 999)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(200, 400)),
        ],
    ),
    pc.ColumnSettings(
        "Platelets",
        impute_type="linear",
        impute_default=300,
        scale_limits=(150, 450),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 1500)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(150, 450)),
        ],
    ),
    pc.ColumnSettings(
        "SOFA",
        scale_limits=(0, 1),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 4)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0, 1)),
        ],
        mapping=compute_SOFA,
    ),
    pc.ColumnSettings(
        "SIRS",
        scale_limits=(0, 2),
        offset=1,
        constraints=[
            pc.ConstraintSettings(PHY_MOD_NAME, limit=(0, 4)),
            pc.ConstraintSettings(NORM_MOD_NAME, limit=(0, 2)),
        ],
        mapping=compute_SIRS,
    ),
]


def add_map_constraint(mod, gvars, ts, k):
    """MAP must be less than 5% different from 2/3 DBP + 1/3 SBP"""
    mod.addConstrs(
        (
            gvars["MAP"][t] <= 1.05 * (2 * gvars["DBP"][t] + gvars["SBP"][t]) / 3
            for t in ts
            if k["MAP"][t] and k["DBP"][t] and k["SBP"][t]
        ),
        "mapsum_ub",
    )
    mod.addConstrs(
        (
            gvars["MAP"][t] >= 0.95 * (2 * gvars["DBP"][t] + gvars["SBP"][t]) / 3
            for t in ts
            if k["MAP"][t] and k["DBP"][t] and k["SBP"][t]
        ),
        "mapsum_lb",
    )


def add_hco3_constraint(mod, gvars, ts, k):
    """If NCO3 is less than 10, BaseExcess must be less than 0"""
    # TODO it seems that the addVars function uses a PyCapsule type somewhere
    # in its implementation, which is unserializable and thus will cause the
    # python multiprocessing library to puke
    switch = mod.addVars(ts, name="nco3_switch", vtype=GRB.BINARY)
    mod.addConstrs(
        (
            10 * switch[t] + 45 * (1 - switch[t]) >= gvars["HCO3"][t]
            for t in ts
            if k["HCO3"][t]
        ),
        "hco3_1",
    )
    mod.addConstrs(
        (10 * (1 - switch[t]) >= gvars["HCO3"][t] for t in ts if k["HCO3"][t]),
        "hco3_2",
    )
    mod.addConstrs(
        (
            gvars["BaseExcess"][t] <= 20 * (1 - switch[t])
            for t in ts
            if k["BaseExcess"][t]
        ),
        "hco3_3",
    )


def add_lactate_constraint(mod, gvars, ts, k):
    """If Lactate is less than 6, BaseExcess must be less than 0"""
    switch = mod.addVars(ts, name="lactate_switch", vtype=GRB.BINARY)
    mod.addConstrs(
        (
            6 * switch[t] + 0 * (1 - switch[t]) <= gvars["Lactate"][t]
            for t in ts
            if k["Lactate"][t]
        ),
        "lactate_1",
    )
    mod.addConstrs(
        (
            6 * (1 - switch[t]) + 30 * switch[t] >= gvars["Lactate"][t]
            for t in ts
            if k["Lactate"][t]
        ),
        "lactate_2",
    )
    mod.addConstrs(
        (
            gvars["BaseExcess"][t] <= 20 * (1 - switch[t])
            for t in ts
            if k["BaseExcess"][t]
        ),
        "lactate_3",
    )


def add_baseexcess_constraint(mod, gvars, ts, k):
    """If BaseExcess is < 0, then either HCO3 is < 10 or Lactate is > 6."""
    switch1 = mod.addVars(ts, name="baseexcess_switch1", vtype=GRB.BINARY)
    switch2 = mod.addVars(ts, name="baseexcess_switch2", vtype=GRB.BINARY)
    switch3 = mod.addVars(ts, name="baseexcess_switch3", vtype=GRB.BINARY)
    mod.addConstrs(
        (
            switch1[t] + 20 * (1 - switch1[t]) >= gvars["BaseExcess"][t]
            for t in ts
            if k["BaseExcess"][t]
        ),
        "baseexcess_1",
    )
    mod.addConstrs(
        (
            -40 * switch1[t] + 0 * (1 - switch1[t]) <= gvars["BaseExcess"][t]
            for t in ts
            if k["BaseExcess"][t]
        ),
        "baseexcess_2",
    )
    mod.addConstrs(
        (
            10 * switch2[t] + 45 * (1 - switch2[t]) >= gvars["HCO3"][t]
            for t in ts
            if k["HCO3"][t]
        ),
        "baseexcess_3",
    )
    mod.addConstrs(
        (
            10 * (1 - switch2[t]) + 0 * switch2[t] <= gvars["HCO3"][t]
            for t in ts
            if k["HCO3"][t]
        ),
        "baseexcess_4",
    )
    mod.addConstrs(
        (
            6 * switch3[t] + 0 * (1 - switch3[t]) <= gvars["Lactate"][t]
            for t in ts
            if k["Lactate"][t]
        ),
        "baseexcess_5",
    )
    mod.addConstrs(
        (
            6 * (1 - switch3[t]) + 30 * switch3[t] >= gvars["Lactate"][t]
            for t in ts
            if k["Lactate"][t]
        ),
        "baseexcess_6",
    )
    mod.addConstrs(
        (switch1[t] <= switch3[t] + switch2[t] for t in ts),
        "baseexcess_7",
    )


def add_ph_constraint(mod, gvars, ts, k):
    """If ph is < 7, then either HCO3 is < 35 or HCO3 is > 10."""
    switch1 = mod.addVars(ts, name="ph_switch1", vtype=GRB.BINARY)
    switch2 = mod.addVars(ts, name="ph_switch2", vtype=GRB.BINARY)
    switch3 = mod.addVars(ts, name="ph_switch3", vtype=GRB.BINARY)
    mod.addConstrs(
        (
            7 * switch1[t] + 7.7 * (1 - switch1[t]) >= gvars["pH"][t]
            for t in ts
            if k["pH"][t]
        ),
        "ph_1",
    )
    mod.addConstrs(
        (
            6.5 * switch1[t] + 7 * (1 - switch1[t]) <= gvars["pH"][t]
            for t in ts
            if k["pH"][t]
        ),
        "ph_2",
    )
    mod.addConstrs(
        (
            35 * switch2[t] + 120 * (1 - switch2[t]) >= gvars["PaCO2"][t]
            for t in ts
            if k["PaCO2"][t]
        ),
        "ph_3",
    )
    mod.addConstrs(
        (
            35 * (1 - switch2[t]) + 16 * switch2[t] <= gvars["PaCO2"][t]
            for t in ts
            if k["PaCO2"][t]
        ),
        "ph_4",
    )
    mod.addConstrs(
        (
            10 * switch3[t] + 45 * (1 - switch3[t]) >= gvars["HCO3"][t]
            for t in ts
            if k["HCO3"][t]
        ),
        "ph_5",
    )
    mod.addConstrs(
        (
            10 * (1 - switch3[t]) + 0 * switch3[t] <= gvars["HCO3"][t]
            for t in ts
            if k["HCO3"][t]
        ),
        "ph_6",
    )
    mod.addConstrs(
        (switch1[t] <= switch3[t] + switch2[t] for t in ts),
        "ph_7",
    )


def add_bilirubin_constraint(mod, gvars, ts, k):
    """Bilirubin_direct must be less than bilirubin total"""
    mod.addConstrs(
        (
            gvars["Bilirubin_direct"][t] <= gvars["Bilirubin_total"][t]
            for t in ts
            if k["Bilirubin_direct"][t] and k["Bilirubin_total"][t]
        ),
        "bilirubin_direct_total",
    )


def add_hgb_constraint(mod, gvars, ts, k):
    """Hct must be less than 1.5 * Hgb"""
    mod.addConstrs(
        (
            gvars["Hct"][t] <= 1.5 * gvars["Hgb"][t]
            for t in ts
            if k["Hct"][t] and k["Hgb"][t]
        ),
        "hct_hgb",
    )


def add_spo2_constraint(mod, gvars, ts, k):
    """SpO2 must be > 90 if SaO2 > 95 at next/previous timepoint"""
    switch1 = mod.addVars(ts, name="spo2_switch1", vtype=GRB.BINARY)
    switch2 = mod.addVars(ts, name="spo2_switch2", vtype=GRB.BINARY)
    ## only apply this constraint for all but the first/last timepoints
    _ts = ts[1:-1]
    mod.addConstrs(
        (0.95 * switch1[t] <= gvars["SaO2"][t - 1] for t in _ts if k["SaO2"][t - 1]),
        "spo2_1",
    )
    mod.addConstrs(
        (0.95 * switch2[t] <= gvars["SaO2"][t + 1] for t in _ts if k["SaO2"][t + 1]),
        "spo2_2",
    )
    mod.addConstrs(
        (
            gvars["O2Sat"][t] >= 0.90 * (switch1[t] + switch2[t] - 1)
            for t in _ts
            if k["O2Sat"][t]
        ),
        "spo2_3",
    )
    mod.addConstrs(
        (
            gvars["SaO2"][t - 1] <= 0.949999 + 0.05 * switch1[t]
            for t in _ts
            if k["SaO2"][t - 1]
        ),
        "spo2_4",
    )
    mod.addConstrs(
        (
            gvars["SaO2"][t + 1] <= 0.949999 + 0.05 * switch2[t]
            for t in _ts
            if k["SaO2"][t + 1]
        ),
        "spo2_5",
    )


def add_custom_constraints(mod, gvars, ts, nulls):
    k = ~nulls
    add_map_constraint(mod, gvars, ts, k)
    add_hco3_constraint(mod, gvars, ts, k)
    add_lactate_constraint(mod, gvars, ts, k)
    add_baseexcess_constraint(mod, gvars, ts, k)
    add_ph_constraint(mod, gvars, ts, k)
    add_bilirubin_constraint(mod, gvars, ts, k)
    add_hgb_constraint(mod, gvars, ts, k)
    add_spo2_constraint(mod, gvars, ts, k)


def path_to_id(path: str) -> str:
    """Given a path to a patient file, return an id."""
    filename = os.path.basename(path)
    parentname = os.path.basename(os.path.dirname(path))
    # ASSUME each patient has the pattern p[0-9]+
    patient_id = os.path.splitext(filename)[0][1:]
    # ASSUME the last letter of the parent directory is a label for the group
    patient_group = parentname[-1]
    return patient_group + patient_id


settings = pc.PipeSettings(
    COLUMNS,
    "SepsisLabel",
    "patient_id",
    separator="|",
    impute=True,
    dirty=True,
    compute=True,
    derive=False,
    scale=True,
    id_val_fun=path_to_id,
    constant_cols=["Age", "Gender"],
    subdf=pc.SubDFSettings(
        12,
        9,
        qp_mods=[
            #pc.QPSettings(
            #    name=PHY_MOD_NAME,
            #    total_diff=True,
            #    diff_matrix=True,
            #    correct=True,
                # custom_constraints=add_custom_constraints,
            #),
            #pc.QPSettings(name=NORM_MOD_NAME, total_diff=True, diff_matrix=True),
        ],
    ),
)
