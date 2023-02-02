#!/usr/bin/env python

from argparse import ArgumentParser

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from pysankey2 import Sankey
from pathlib import Path
from typing import Union

import os.path as op
import pandas as pd
import numpy as np
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


atlas_lut = {
    "des": "Desikan",
    "aal": "AAL",
    "hox": "Harvard-Oxford",
    "cc2": "Craddock 200"
}
mpl.rcParams.update({"font.size": 26, "axes.linewidth": 4})

def load_and_stack(bp: Union[str, Path], pattern: str=None) -> pd.DataFrame:
    # Find, load, and stack all relevant dfs
    df_dict = {_: pd.read_hdf(_) for _ in bp.rglob(pattern)}

    df = None
    # For each dataframe, annotate with dataset and atlas, then stack
    for k, v in df_dict.items():
        dset, atlas = tuple(k.name.split('_')[0:2])
        v['dataset'] = dset
        v['atlas'] = atlas
        df = pd.concat([df, v])

    return df


def reindex_clusters(values: np.array, order: np.array=np.empty(0)) \
                                                          -> np.array:
    # Bumps larger clusters to start of list, or predefined order
    labels, counts = np.unique(values, return_counts=True)
    # If the new ordering is empty, resort values based on N-occurances
    if order.size == 0:
        order = np.argsort(np.argsort(counts)[::-1])
    
    # To be extra careful with overwriting, create new, empty, list
    new_values = np.empty_like(values)
    for oldx, newdx in zip(np.sort(labels), order):
        # Re-index according to the sorting order
        new_values[np.where(values == oldx)] = newdx
    
    return new_values


def cluster_subjects(df: pd.DataFrame, dmat: np.array, ofile: Path,
                     anno: bool=True) \
                                               -> (np.array, pd.DataFrame):
    # Takes datadrame and a distance matrix, and returns clusterings
    
    # Invert normalized distance matrix to be interpretted as similarity matrix
    smat = 1 - dmat
    
    # Define clustering... N.B. 90-percentile cut-off was found empirically
    lim = np.percentile(smat, 90)
    clf = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                  linkage='complete', distance_threshold=lim)

    # Fit clustering, and re-sort clusters according to size, and grab counts
    values = clf.fit_predict(smat)
    values = reindex_clusters(values)
    labels, counts = np.unique(values, return_counts=True)

    # Extract and plot cluster signatures
    fig = plt.figure(figsize=(len(labels)*6, 6))

    clustering = []
    for _, (l, c) in enumerate(zip(labels, counts)):
        locs = np.where(values == l)[0]
        tdf = df.iloc[locs]
        signature = np.mean(tdf['rank_corr'])
        clustering += [{
            'label': l,
            'members': locs,
            'signature': signature
        }]

        plt.subplot(1, len(labels), _+1)
        plt.imshow(signature, cmap='cividis')
        plt.yticks([])
        plt.xticks([])
        if anno:
            plt.xlabel('Label: {0}  |  N: {1}'.format(l, len(locs)))

    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()
    return values, clustering


def plot_clustered_dset(dmat: np.array, label_list: np.array, ofile: Path,
                                                           title: str) -> None:
    cluster_sizes = [len(_) for _ in label_list]
    start_points = [sum(cluster_sizes[:_]) for _ in range(len(cluster_sizes))]

    order = np.concatenate(label_list)
    resorted = dmat[:, order][order]
    
    f = plt.figure(figsize=(15, 15))
    ax = f.add_subplot(111)
    plt.imshow(resorted, cmap='inferno')

    for s, l in zip(start_points, cluster_sizes):
        rect = patches.Rectangle((s-.5, s-0.5), l, l, linewidth=6,
                                 edgecolor='w', facecolor='none', zorder=10)
        ax.add_patch(rect)

    plt.yticks([])
    plt.xticks([])
    plt.title(title)
    plt.tight_layout()

    plt.savefig(ofile)
    plt.close()


def plot_all_difference_matrices(df: pd.DataFrame, ofile: Path):

    n_cols = 10
    n_rows = int(len(df) / n_cols) + 1

    fig = plt.figure(figsize=(n_cols*2, n_rows*2))

    for idx, row in df.iterrows():
        fig.add_subplot(n_rows, n_cols, idx+1)
        plt.imshow(row.rank_corr)
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()


def single_study_eval(df_subs: pd.DataFrame, df_dist: pd.DataFrame,
                                                        odir: Path) -> None:
    # Do clustering, plotting, & similarity computation for each dataset

    combinations = df_subs.value_counts(['dataset', 'atlas']).index

    # Define convenience slicing lambda
    slice_da = lambda _df, dset, atlas: _df[(_df['dataset'] == dset) &
                                            (_df['atlas'] == atlas)]
    membership = []
    clusters = []

    # For each dataset and atlas...
    for ds, at in combinations:
        print(ds, at)

        # Grab the similarity matrices and distance matrices for current ds, at
        t_subs = slice_da(df_subs, ds, at)
        t_dist = slice_da(df_dist, ds, at)['distance'].values[0]

        # Prepare metadata to help with plotting & storage
        att = atlas_lut[at]
        title = "{0} — {1} Parcellation".format(ds, att)
        t_odir = odir / ds
        t_odir.mkdir(parents=True, exist_ok=True)
        t_ofile1 = t_odir / "{0}_signature.pdf".format(at)
        t_ofile2 = t_odir / "{0}_signature_all.pdf".format(at)

        # Cluster & plot cluster signatures
        t_labels, t_clust = cluster_subjects(t_subs, t_dist, t_ofile1)
        # TODO: make the below not crash for large images
        # plot_all_difference_matrices(t_subs, t_ofile2)

        # Extract labels and prepare metadata for dataset-wide plotting
        label_list = [_['members'] for _ in t_clust]
        mn, sd = np.mean(t_dist), np.std(t_dist)
        title = "{0}\n({1:.2f} ± {2:.2f})".format(title, mn, sd)
        t_ofile = t_odir / "{0}_clustering.pdf".format(at)

        # Plot clustered subject signatures
        plot_clustered_dset(t_dist, label_list, t_ofile, title=title)

        # Create convenience table of clustering results
        membership += [{
            "dataset": ds,
            "atlas": at,
            "labels": t_labels,
        }]
        
        # Add dataset/atlas annotations to cluster data, and store
        t_clust = pd.DataFrame.from_dict(t_clust)
        t_clust["dataset"] = ds
        t_clust["atlas"] = at
        clusters += [ t_clust ]

    # Return a single dataframe for each of the two storage containers
    return pd.DataFrame.from_dict(membership), pd.concat(clusters)


def main():
    parser = ArgumentParser()
    parser.add_argument("dset", help="Path to dataset director(y/ies)")
    parser.add_argument("outdir", help="Path to output directory")

    results = parser.parse_args()

    dset = Path(results.dset)
    odir = Path(results.outdir)

    # Load dataframes and distance matrices
    # TODO: update up-stream script to use pickle instead of hdf5
    df_subs = load_and_stack(dset, pattern="*similarity*h5")
    dists = [{ 'dataset': _.name.split('_')[0],
               'atlas':_.name.split('_')[1],
               'distance': np.loadtxt(_) }
             for _ in dset.rglob('*distmat*txt')]
    df_dist = pd.DataFrame.from_dict(dists)

    # Save clustered data to disk
    membership, clustering = single_study_eval(df_subs, df_dist, odir)
    membership.to_pickle(odir / "cluster_membership.pkl")
    clustering.to_pickle(odir / "cluster_definitions.pkl")


if __name__ == "__main__":
    main()

