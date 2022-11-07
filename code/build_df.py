#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np


atlas_lut = {
    'cc2': 200,
    'des': 70,
    'aal': 116,
    'hox': 54,
}

schema = pa.schema({
    'dataset': pa.string(),
    'subject': pa.string(),
    'session': pa.string(),
    'alignment': pa.string(),
    'filtering': pa.bool_(),
    'scrubbing': pa.bool_(),
    'gsr': pa.bool_(),
    'parcellation': pa.string(),
    'pipeline': pa.string(),
    'graph': pa.array(),
    'path': pa.string()
})

def extract_info(path: Path, N: int) -> dict:
    sub = path.stem.split('_')
    par = path.parent.stem
    df = {
        'dataset': sub[0],
        'subject': sub[1],
        'session': sub[3],
        'alignment': 'FNIRT' if "FSL" in par else "ANTs",
        'filtering': True if 'frf' in par else False,
        'scrubbing': True if 'scr' in par else False,
        'gsr': True if 'gsr' in par else False,
        'parcellation': par.split('_')[-1],
        'pipeline': par,
        'graph': grab_mat(path, N),
        'path': str(path)
    }
    return df


def dummy_loadtxt(path: Path) -> np.array:
    try:
        return np.loadtxt(path)
    except ValueError as e:
        with open(path, 'r') as fhandle:
            edges = fhandle.readlines()
            return np.array([tuple(e.split()) 
                             if len(e.split() == 3)
                             else tuple(*e.split(), 0)
                             for e in edges]).astype(float)


def grab_mat(path: Path, N: int, dtype:type = np.float32) -> np.array:
    elist = dummy_loadtxt(path)
    m = np.zeros((N, N))
    for e in range(elist.shape[0]):
        m[int(elist[e,0]), int(elist[e,1])] = elist[e,2]
    locs = np.triu_indices(N)
    return m[locs].astype(dtype)


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("output_loc")
    parser.add_argument("--parcellation", "-p", default='aal',
                        choices=['cc2', 'hox', 'aal', 'des'])

    # Basic parameter setup
    results = parser.parse_args()
    dset = results.dataset_path
    parc = results.parcellation

    # Grab filelist & extract info
    fls = list(Path(dset).rglob("*.ssv"))

    N = atlas_lut[parc]
    df_dict = [extract_info(f, N) for f in fls
               if f.parent.match("*{0}*".format(parc))]

    # Write to file
    df = pa.Table.from_pylist(df_dict,schema=schema)
    ofname = results.output_loc + ".parquet"
    pq.write_table(df, ofname, compression='gzip')


if __name__ == "__main__":
    main()
