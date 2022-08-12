import os
import sys
from textwrap import indent
from math import sqrt, log
from typing import List, Optional, Tuple, Dict, Generator, Callable, Any, NamedTuple
from contextlib import contextmanager
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

"""
Settings classes

The PIPE algorithm is configured using one read-only data structure. This is
represented as a hierarchy of subclassed Named Tuples (inherently immutable).

It is expected that the end user assemble these classes to represent their
problem, and is kept as general as possible to separate the code logic from any
domain specific concerns.

Note that because these are inherently immutable, the PIPE algorithm is
effectively stateless.
"""

_Limit = Tuple[float, float]


class ConstraintSettings(NamedTuple):
    """A class representing constraints applied to an individual column.

    Note that this object represents a shorthand for defining constraints at the
    column level. For constraints requiring more than that defined here and/or
    constraints between multiple columns, see custom_constraints in the
    QPSettings class.

    Attributes:
    ---
    mod_name: the name of the quadratic programming model to which this
        constraint belongs (see the attribute name in QPSettings, must match
        exactly)

    limit: a tuple like (LOWER, UPPER) representing a constraint like
        "LOWER <= VALUE <= UPPER" where VALUE is any given value in the column

    offset_limit: a tuple like (LOWER, UPPER) representing a constraint like
        "LOWER <= VALUE0 - VALUE1 <= UPPER" where VALUE0 is the column value at
        a given timepoint and VALUE1 is another value some number of units into
        the future (where the number of units is given by the offset attribute)

    offset: represents the offset for the offset_limit parameter
    """

    mod_name: str
    limit: Optional[_Limit] = None
    offset: int = 1
    offset_limit: Optional[_Limit] = None


class ColumnSettings(NamedTuple):
    """A class representing how to process a column in a dataframe/sub-dataframe.

    Attributes:
    ---
    name: the name of the column (must match a column header exactly in the
        input dataframe (unless mapping is defined; see below)

    impute_type: a string representing how to impute missing values (if needed);
        for now only "linear" and "stepwise" are valid, which correspond to
        linear interpolation and fill-forward interpolation respectively

    impute_default: the default value when imputing fails; in the case of fill
        forward interpolation, this will be used to fill in any initial missing
        values before the first non-missing value; in the case of any
        interpolation method the entire column will take this value if it is
        fully empty

    constraints: a list of ConstraintSettings objects

    scale_limits: a tuple like (LOWERBOUND, UPPERBOUND) where the column will be
        linearly scaled such that LOWERBOUND will be 0 and UPPERBOUND will be 1;
        note that this applied to the original column and not the
        log-transformed column in the case the log_transform is True

    log_transform: if True, transform the column using the function
        log10(VALUE + 1)

    mapping: a function to compute this column from other columns; this function
        expects a pandas.DataFrame and should return a pandas.Series; this
        series will be added to a new column denoted by the name attribute

    offset: the offset to use when calculating derivatives

    Methods:
    ---
    qp_mod_names: return a list of the quadratic programming model names from
        then constraints attribute
    """

    name: str
    impute_type: Optional[str] = None
    impute_default: Optional[float] = None
    constraints: List[ConstraintSettings] = []
    scale_limits: Optional[_Limit] = None
    log_transform: bool = False
    transform: Optional[Callable[[float], float]] = None
    mapping: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    offset: Optional[int] = None

    @property
    def qp_mod_names(self):
        return [c.mod_name for c in self.constraints]


class QPSettings(NamedTuple):
    """A class representing how to compute a quadratic programming (QP) model.

    For a given sub-dataframe S, the QP model can compute a new sub-dataframe S'
    representing S constrained to the feasible region defined by the constraints
    of the QP models (see function qp_mod).

    Once S' is computed, several options are possible:
    1. correction: overwrite the values of S with S' (eg if any given value is
       outside the feasible region, replace it with the nearest point on the
       feasible region)
    2. adding difference columns: add new columns computed by S - S' (eg the
       distance of each column in the sub-dataframe from the feasible region)
    3. add total difference: append the total distance between the original
       subdataframe and the feasible region; this is computed as
       sqrt((S - S')^2.sum().sum()) (euclidean distance)

    NOTE: the constraints are partly defined here and partly defined column-wise
    via the ColumnSettings object.

    Class attributes:
    ---
    name: the name used to reference the model

    correct: if True, correct via (1) above

    diff_matrix: if True, add the difference matrix as described in (2) above

    stats: a list of statistics to calculate; this is analogous to the same
        option as that in SubDFSettings; here the statistics will be applied to
        the difference columns as computed via (2) above

    total_diff: if True, add the total difference for this QP model as defined
        by (3) above

    custom_constraint: a function that will apply custom constraints to the
        model; this function takes 4 arguments: the Gurobi model object, a list
        of Gurobi variables, a list of timestamps for the subdataframe, and a
        boolean matrix corresponding to the subpatient dataframe where the
        True represents that a value is missing and False otherwise (see fit_qp
        for how this function is called); this function should return None

    Methods:
    ---
    total_diff_column: the name of the column to be used when storing the
        total difference value as defined in (3) above; will be formatted like
        total_MODNAME_diff where modname corresponds to the name attribute.
    """

    name: str
    correct: bool = False
    stats: List[str] = []
    total_diff: bool = False
    diff_matrix: bool = False
    custom_constraints: Optional[Callable[[Any, Any, Any, Any], None]] = None

    @property
    def total_diff_column(self):
        if self.total_diff:
            return "total_{}_diff".format(self.name)
        else:
            return ""


class SubDFSettings(NamedTuple):
    """A class representing the settings for processing sub-dataframes.

    Since PIPE deals with time-series data which may have an arbitrary number of
    rows, one way to "standardize" each dataframe is to break it apart into
    pieces of fixed size with some overlap between each piece. Each piece is
    called a "sub-dataframe" here.

    Class attributes:
    ---
    width: the width one subdataframe

    overlap: the overlap of each subdataframe (0 = no overlap)

    stats: a list of statistics to calculate for each column in the
        subdataframe; this should be a list that can be consumed by
        pandas.DataFrame.aggregate

    derivative_stats: like the stats attribute but for derivative columns (if
       included)

    qp_mods: a list of QPSettings describing quadratic programming models to use
        when processing each subdataframe; each model will be applied in the
        order shown by this list
    """

    width: int
    overlap: int
    stats: List[str] = []
    derivative_stats: List[str] = []
    qp_mods: List[QPSettings] = []


class PipeSettings(NamedTuple):
    """A class representing the settings with which to run PIPE.

    Class Attributes:
    ---
    columns_settings: a list of ColumnSettings objects corresponding to the
      independent variables in the dataframe to process

    response_col: the name of the dependent variable; must correspond to a
      column in the input dataframe

    id_col: the name of the id column to use; this does not need to correspond
        to a column in the dataframe as it will be added

    separator: the separator to use when parsing the input file to a dataframe

    subdf: a SubDFSettings object indicating how to process each sub-dataframe

    impute: if True, apply imputation (imputation method is defined per-column
        via the column_settings attribute)

    scale: if True, apply scaling and/or log-transforms (scaling is defined
        per-column via the column_settings attribute)

    derive: if True, calculate derivatives for each column

    compute: if True, add "computed" columns (these are defined on a per-column
        basis via the column_settings attribute)

    dirty: if True, add dirty-bit columns (1 if a column is empty, 0 otherwise)

    constant_cols: a list of columns which are "constant" (eg they are not used
        in the computation but will be included in the output)

    id_val_fun: a function that takes a path and returns a string to use for
        id column (named via the id_col attribute); by default this takes the
        basename of the filepath

    Methods
    ---
    column_names: return a list of all column names as defined by the
        column_settings attribute

    column_settings_noncompute: return a list of all ColumnSettings objects
        without those without a mapping function

    column_names_explicit: the list of column names for the columns returned by
        column_settings_noncompute

    qp_column_names: return a list of column names corresponding to the new
        columns created when applying the quadratic programming models defined
        in SubDFSettings; these will be formatted like "MODNAME_COLNAME_diff"
        where MODNAME is the name of the quadratic programming model and COLNAME
        is the column name

    column_names_derived: return a list of column names corresponding to the
        derivative columns to be added; only columns with the offset attribute
        will be included in this
    """

    column_settings: List[ColumnSettings]
    response_col: str
    id_col: str
    separator: str = ","
    subdf: Optional[SubDFSettings] = None
    constant_cols: List[str] = []
    id_val_fun: Callable[[str], str] = lambda p: os.path.basename(p)
    impute: bool = False
    scale: bool = False
    dirty: bool = False
    derive: bool = False
    compute: bool = False

    # TODO enforce any dependencies and restrictions (eg we shouldn't have
    # it compute the qp model within a subdf and for the whole patient
    # TODO ensure columns that contain constraints refer to models that
    # exist
    @property
    def column_names(self):
        return [s.name for s in self.column_settings]

    @property
    def column_settings_noncompute(self):
        return (
            self.column_settings
            if self.compute is False
            else [s for s in self.column_settings if s.mapping is None]
        )

    @property
    def column_names_explicit(self):
        return [s.name for s in self.column_settings_noncompute]

    @property
    def qp_column_names(self):
        return (
            {}
            if self.subdf is None
            else {
                mod_name: {
                    s.name: "{}_{}_diff".format(mod_name, s.name)
                    for s in self.column_settings
                    if mod_name in s.qp_mod_names
                }
                for mod_name in [m.name for m in self.subdf.qp_mods]
            }
        )

    @property
    def column_names_derived(self):
        return [s.name for s in self.column_settings if s.offset is not None]


"""
Quadratic Programming Functions
"""


@contextmanager
def _suppress_stdout() -> Generator:
    """Generate a context to run functions with stdout suppressed."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _get_var_name(v) -> str:
    """Given a variable from Gurobi, return its name."""
    i = v.varName.find("[")
    return v.varName[0:i]


def _optimal_vars_to_df(column_height: int, mod: gp.Model, nulls) -> pd.DataFrame:
    """Return a dataframe representing the optimized matrix from Gurobi."""
    out = mod.getVars()
    rep = {}
    for i in range(0, len(out), column_height):
        n = _get_var_name(out[i])
        k = ~nulls[n]
        u = i + column_height
        rep[n] = [c.X if k.iloc[I] else np.nan for I, c in enumerate(out[i:u])]
    return pd.DataFrame.from_dict(rep)


def _fit_qp(
    df: pd.DataFrame,
    settings: PipeSettings,
    qp_settings: QPSettings,
) -> pd.DataFrame:
    """Fit a quadratic programming model.

    For a given M*N dataframe D, define variable objects in Gurobi corresponding
    to each cell: D[m][n] -> V[m][n]. Then define an objective function such
    that sum((V[1][1] - D[1][1])^2 + ... + (V[m][n] - D[m][n])^2) should be
    minimized.

    Constraints for the model will be applied from two sources: the columns (see
    the constraints parameter in the ColumnSettings class) or the
    custom_constraints parameter defined in the QPSettings class. These are
    accessed using the settings and qp_settings function parameter respectfully
    here.

    Note that constraints will only be added if all corresponding cells in
    the dataframe are not NaNs (null). This is enforced here for any constraints
    defined via ColumnSettings, and is the user's responsibility to enforce when
    passing the custom_constraints function to the QPSettings class. Obviously
    none of this matters if imputation is to be performed on the dataframe, but
    this guarantees success in case this is false.
    """
    mod_name = qp_settings.name
    params = [*settings.qp_column_names[mod_name]]
    ts = df.index.tolist()
    mod = gp.Model("qp")
    # make lower bound some stupidly large negative number because gurobi
    # doesn't seem to like GRB.INFINITY
    qp_vars = {x: mod.addVars(ts, name=x, lb=-10000000) for x in params}
    nulls = df.isnull()

    # add constraints: for all columns, find constraints pertaining to the
    # given model name and add limits and offset limits if applicable
    for c in [c for c in settings.column_settings if c.name in params]:
        n = c.name
        v = qp_vars[n]
        k = ~nulls[n]
        for cn in [cn for cn in c.constraints if cn.mod_name == mod_name]:
            if cn.limit:
                L = cn.limit
                mod.addConstrs(
                    (v[t] >= L[0] for t in ts if k[t]),
                    n + "_lb",
                )
                mod.addConstrs(
                    (v[t] <= L[1] for t in ts if k[t]),
                    n + "_ub",
                )
            if cn.offset_limit:
                L = cn.offset_limit
                o = cn.offset
                mod.addConstrs(
                    (v[t] - v[t + o] >= L[0] for t in ts[:-o] if k[t] and k[t + o]),
                    n + "_cub",
                )
                mod.addConstrs(
                    (v[t] - v[t + o] <= L[1] for t in ts[:-o] if k[t] and k[t + o]),
                    n + "_clb",
                )

    ## add custom constraints
    custom = qp_settings.custom_constraints
    if custom is not None:
        custom(mod, qp_vars, ts, nulls)

    # objective function: for all X at all t: min sum[(X(t)-X0(t))^2] where X0
    # is the value for X at t in the original dataset
    obj = 0  # dummy to prime python variable before the loop (yay monoids...)
    for p in params:
        for t in ts[0:]:
            if not nulls[p][t]:
                obj += (qp_vars[p][t] - df[p][t]) * (qp_vars[p][t] - df[p][t])

    mod.setObjective(obj, GRB.MINIMIZE)

    # hide annoying output when optimizing
    with _suppress_stdout():
        mod.optimize()

    return _optimal_vars_to_df(len(df), mod, nulls)


"""
Sub-dataframe functions
"""


def _get_rolling_slices(n: int, width: int, overlap: int) -> List[Tuple[int, int]]:
    """Return indices with which to slice a dataframe to sub-dataframes."""
    uppers = [u for u in list(range(n, 0, -overlap))]
    s = [(u - width, u) for u in uppers if u - width >= 0]
    s.reverse()
    return s


def _flatten_indexed(df: pd.DataFrame) -> pd.Series:
    """Flatten a dataframe to a vector.

    The index of the vector (actually a pandas.Series object) will be formatted
    like COLNAME-ROW where COLNAME is the name of the column and ROW is the
    number of the row the corresponding value was located within the original
    dataframe.
    """
    _df = df.transpose().stack()
    _df.index = ["{}-{}".format(i, n) for i, n in _df.index.tolist()]
    return _df


def _flatten_statistics(stats: List[str], df: pd.DataFrame) -> pd.Series:
    """Like _flatten_indexed but calculate statistics first."""
    return _flatten_indexed(df.aggregate(stats))


def _compute_derivatives(df: pd.DataFrame, settings: PipeSettings) -> pd.DataFrame:
    """Return a dataframe with derivative columns appended."""
    _df = pd.DataFrame()
    for s in settings.column_settings:
        if s.offset is not None:
            _df["{}_derivative".format(s.name)] = df[s.name].diff(s.offset).fillna(0)
    return _df


def _apply_scaling(df: pd.DataFrame, settings: PipeSettings) -> pd.DataFrame:
    """Scale and apply log-transforms to a dataframe."""
    for c in settings.column_settings:
        if c.scale_limits is not None:
            ser = df[c.name]
            lower, upper = c.scale_limits
            if c.log_transform:
                log_lower = log(lower + 1, 10)
                log_upper = log(upper + 1, 10)
                df[c.name] = (np.log10(ser + 1) - log_lower) / (log_upper - log_lower)
            else:
                df[c.name] = (ser - lower) / (upper - lower)
        elif c.log_transform is True:
            df[c.name] = np.log10(df[c.name] + 1)
    return df


def _get_subdf_vector(
    df: pd.DataFrame, all_qp_settings: List[QPSettings], settings
) -> pd.Series:
    """Return a flattened vector corresponding to a sub-dataframe."""
    diff_vectors: List[pd.Dataframe] = []
    diff_stats_vectors: List[pd.Dataframe] = []
    total_diffs: Dict[str, float] = {}
    _df = df[settings.column_names].reset_index(drop=True)
    for qp_settings in all_qp_settings:
        _optimized_df = _fit_qp(_df, settings, qp_settings)
        if (
            qp_settings.diff_matrix is True
            or qp_settings.total_diff is True
            or len(qp_settings.stats) > 0
        ):
            ## TODO these copy() calls seem stupid
            _diff_matrix = (
                _optimized_df - _df
                if settings.scale is False
                else _apply_scaling(_optimized_df.copy(), settings)
                - _apply_scaling(_df.copy(), settings)
            )
            mapper = settings.qp_column_names[qp_settings.name]
            _diff_matrix.columns = [mapper[n] for n in _diff_matrix.columns.tolist()]
            if qp_settings.diff_matrix is True:
                diff_vectors = diff_vectors + [_flatten_indexed(_diff_matrix)]
            if len(qp_settings.stats) > 0:
                diff_stats_vectors = diff_stats_vectors + [
                    _flatten_statistics(qp_settings.stats, _diff_matrix)
                ]
            if qp_settings.total_diff is True:
                _total_diff = sqrt(_diff_matrix.pow(2).sum().sum())
                total_diffs[qp_settings.total_diff_column] = _total_diff
        if qp_settings.correct is True:
            _df = _optimized_df
    if settings.scale is True:
        _df = _apply_scaling(_df, settings)
    ## add data columns
    data_vector = _flatten_indexed(_df)
    stats_vector = (
        []
        if len(settings.subdf.stats) == 0
        else [_flatten_statistics(settings.subdf.stats, _df)]
    )
    total_diff_vector = (
        []
        if len(total_diffs) == 0
        else [pd.Series([*total_diffs.values()], index=[*total_diffs])]
    )
    ## add derivative columns
    if settings.derive is True:
        _df_derivative = _compute_derivatives(_df, settings)
    derivative_vectors = (
        [] if settings.derive is False else [_flatten_indexed(_df_derivative)]
    )
    derivative_stats_vectors = (
        []
        if len(settings.subdf.derivative_stats) == 0
        else [_flatten_statistics(settings.subdf.derivative_stats, _df_derivative)]
    )
    response = df[settings.response_col].max()
    return pd.concat(
        [data_vector]
        + stats_vector
        + total_diff_vector
        + diff_vectors
        + diff_stats_vectors
        + derivative_vectors
        + derivative_stats_vectors
        + [pd.Series([response], index=[settings.response_col])]
    )


def _get_subdf_matrix(
    df: pd.DataFrame,
    pid: str,
    constant_col_values: dict,
    settings: PipeSettings,
    sp_settings: SubDFSettings,
) -> pd.DataFrame:
    """Break dataframe into sub-dataframes and process.

    High level steps for this function:
    1. slice dataframe into subdataframes
    2. process each sub-dataframe (eg apply QP models, scale, apply stats, etc)
    3. flatten each sub-dataframe into a vector
    4. concat all vectors vertically into a new dataframe
    """
    return (
        pd.concat(
            (
                _get_subdf_vector(df[l:u], sp_settings.qp_mods, settings)
                for l, u in _get_rolling_slices(
                    len(df), sp_settings.width, sp_settings.overlap
                )
            ),
            axis=1,
        )
        .transpose()
        .assign(**{settings.id_col: pid})
        .assign(**constant_col_values)
    )


"""
Sub-dataframe pre-processing functions
"""


def _get_dirty_dict(df, settings: PipeSettings) -> Dict[str, int]:
    """Return a dict representing which columns are dirty.

    "Dirty" means that all values are NaNs. The keys in the return dict will
    be column names, and the values will be 1 if dirty and 0 otherwise.
    """
    return {
        name + "_dirty": value
        for name, value in df.drop(columns=[settings.id_col, settings.response_col])
        .isnull()
        .all()
        .astype(int)
        .to_dict()
        .items()
    }


def _impute_columns(df, settings: PipeSettings):
    """Apply imputation to a dataframe."""
    # TODO this is all nice but maybe add another column to track all cells
    # that were imputed
    for s in settings.column_settings:
        n = s.name
        t = s.impute_type
        d = s.impute_default
        if t is not None:
            if t == "linear":
                df[n] = df[n].interpolate(method="linear").fillna(method="bfill")
            elif t == "stepwise":
                df[n] = df[n].interpolate(method="pad")
            else:
                print("Warning: unknown inpute type: {}".format(t))
            df[n] = df[n].fillna(d)
    return df


def _compute_columns(df, settings):
    """Add computed columns to dataframe.

    Computed column means a column that does not exist in the original dataframe
    that is computed from columns that do exist.

    See the mapping attribute of the ColumnSettings class for how create a
    computed column.
    """
    for cs in settings.column_settings:
        if cs.mapping is not None:
            df[cs.name] = cs.mapping(df)
    return df


"""
Main entrypoint
"""


def process_df(settings: PipeSettings, path: str):
    """Process a dataframe using PIPE.

    Return a processed dataframe.

    Parameters:
    ---
    path: a path to the file encoding the dataframe to process

    settings: an PipeSettings object (see this class for details)
    """
    pid = settings.id_val_fun(path)
    df = pd.read_csv(path, sep=settings.separator)

    # if we wish to return a subdf matrix, return None if the patient has
    # less rows than window width
    if settings.subdf is not None and len(df) < settings.subdf.width:
        return None

    constant_col_values = {k: df[k].iloc[0] for k in settings.constant_cols}

    df = df.loc[:, settings.column_names_explicit + [settings.response_col]].assign(
        **{settings.id_col: pid}
    )

    if settings.dirty is True:
        dirty_dict = _get_dirty_dict(df, settings)

    if settings.impute is True:
        df = _impute_columns(df, settings)

    if settings.compute is True:
        df = _compute_columns(df, settings)

    # TODO add qp_mods outside the subdf part?

    if settings.subdf is not None:
        df = _get_subdf_matrix(df, pid, constant_col_values, settings, settings.subdf)
    else:
        ## TODO This seems redundant
        df = df.assign(**constant_col_values)
        if settings.scale is True:
            df = _apply_scaling(df, settings)

    if settings.dirty is True:
        df = df.assign(**dirty_dict)

    return df
