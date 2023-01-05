#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import networkx as nx
import numpy as np
import os


atlas_lut = {
    'cc2': 200,
    'des': 70,
    'aal': 116,
    'hox': 54,
}

# TODO: figure out how to include arrays in the schema
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
#     'graph': pa.array(),
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
                             if len(e.split()) == 3
                             else tuple([*e.split(), 0])
                             for e in edges]).astype(float)


def grab_mat(path: Path, N: int, dtype:type = np.float32) -> np.array:
    locs = np.triu_indices(N)
    if path.suffix == '.ssv':
        elist = dummy_loadtxt(path)
        m = np.zeros((N, N))
        for e in range(elist.shape[0]):
            m[int(elist[e,0]), int(elist[e,1])] = elist[e,2]
    elif path.suffix == '.graphml':
        g = nx.read_graphml(path)
        m = nx.to_numpy_array(g)
        del g
    return m[locs].astype(dtype)


def search_and_construct(dset: Path, dname: str, output: Path,
                         parcellation: str="aal",
                         extension: str="graphml") -> Path:
    # Grab filelist & extract info
    fls = list(Path(dset).rglob("*.{0}".format(extension)))

    N = atlas_lut[parcellation]
    df_dict = [extract_info(f, N) for f in fls]

    # Write to file
    df = pa.Table.from_pylist(df_dict)
    ofname = output / (str(dset.stem) + ".parquet")
    print(ofname)
    pq.write_table(df, ofname, compression='gzip')
    return ofname


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("output_path")
    parser.add_argument("--parcellation", "-p", default='aal',
                        choices=['cc2', 'hox', 'aal', 'des'])
    parser.add_argument("--extension", "-e", default='graphml',
                        choices=['graphml', 'ssv'])

    # Basic parameter setup
    results = parser.parse_args()
    dset = Path(results.dataset_path)
    oloc = Path(results.output_path)
    parc = results.parcellation
    ext = results.extension

    futures, dbpaths = [], []

    with ProcessPoolExecutor(max_workers=8) as pool:
        for d in os.listdir(dset):
            if d.endswith('tgz'):
                continue
            for p in os.listdir(dset / d / "graphs"):
                if p.endswith('.rds') or parc not in p:
                    continue
                datadir = dset / d / "graphs" / p
                outdir = oloc / d
                outdir.mkdir(exist_ok = True)

                futures.append(pool.submit(search_and_construct, datadir,
                                           d, outdir, parc, ext))

    for future in as_completed(futures):
        dbpaths.append(future.result())

    print(dbpaths)


if __name__ == "__main__":
    main()
