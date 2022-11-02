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
    'aal': 108,
    'hox': 54,
}

def extract_info(path: Path, N: int):
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


def grab_mat(path: Path, N: int, dtype:type = np.float32):
    try:
        elist = np.loadtxt(path)
        m = np.zeros((N, N))
        for e in range(elist.shape[0]):
            m[int(elist[e,0]), int(elist[e,1])] = elist[e,2]
        locs = np.triu_indices(N)
        return m[locs].astype(dtype)
    except ValueError as e:
        return None


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("output_loc")
    parser.add_argument("--parcellation", "-p", default='cc2',
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
    df = pa.Table.from_pandas(pd.DataFrame.from_dict(df_dict),
                              preserve_index=False)
    ofname = results.output_loc + ".parquet"
    pq.write_table(df, ofname, compression='gzip')


if __name__ == "__main__":
    main()
