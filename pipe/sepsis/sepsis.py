import pipe.core.core as pc
import pandas as pd
import time
import os.path
import multiprocessing as mp
from functools import partial
from typing import List, Optional

"""
Sepsis processing pipeline.

This will process all physionet patient files using PIPE and concatenate them
into a giant dataframe for further downstream processing.
"""


def _list_relative_files(dirs: List[str], limit: Optional[int] = None) -> List[str]:
    """Given a list of directories, return relative paths for all files inside."""
    ps = [os.path.join(d, f) for d in dirs for f in os.listdir(d)]
    ps.sort()
    return ps[:limit] if limit else ps


def import_all(
    settings: pc.PipeSettings,
    datadirs: List[str],
    multicore: bool = True,
    limit: int = None,
) -> pd.DataFrame:
    """Return a dataframe for all patients processed using PIPE.

    In addition to the PipeSettings object and the list of directories to
    process, the behavior of this can be controlled using the following:

    multicore: the number of cores to use

    limit: the maximum number of files to process (useful for small test runs)
    """

    paths = _list_relative_files(datadirs, limit)

    print("Importing {} patient files".format(len(paths)))

    f = partial(pc.process_df, settings)

    if multicore:
        # I'm impatient, make my cpu scream :)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            dfs = list(pool.map(f, paths, 10))
    else:
        dfs = list(map(f, paths))

    print("Concatenating patient files")
    return pd.concat(dfs)


def write_summary_data(
    df: pd.DataFrame, settings: pc.PipeSettings, resultsdir: str
) -> None:
    """Write a processed dataframe and the settings to disk."""
    print("Writing patient summary data to csv")
    ut = int(time.time())
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    summary_outpath = os.path.join(resultsdir, "summary_{}.csv".format(ut))
    setting_outpath = os.path.join(resultsdir, "settings_{}.txt".format(ut))
    df.to_csv(summary_outpath)
    with open(setting_outpath, "w") as f:
        f.write(str(settings))
