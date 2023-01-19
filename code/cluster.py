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


def cluster_subjects(df: pd.DataFrame, dmat: np.array, anno: bool=False,
                            plot_title: str='') -> (np.array, pd.DataFrame):
    # Takes datadrame and a distance matrix, and returns clusterings
    
    # Invert normalized distance matrix to be interpretted as similarity matrix
    smat = 1 - dmat
    
    # Define clustering... N.B. 90-percentile cut-off was found empirically
    lim = np.percentile(smat, 90)
    clf = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                  linkage='complete', distance_threshold=lim)

    # Fit clustering, and re-sort clusters according to size, and grab counts
    values = clf.fit_predict(smat)
    print(values)
    values = reindex_clusters(values)
    labels, counts = np.unique(values, return_counts=True)

    # Extract and plot cluster signatures
    fig = plt.figure(figsize=(20, len(labels)*5))

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
            plt.title(plot_title)

    plt.show()
    return values, clustering


def plot_clustered_dset(dmat: np.array, label_list: np.array) -> None:
    cluster_sizes = [len(_) for _ in label_list]
    start_points = [sum(cluster_sizes[:_]) for _ in range(len(cluster_sizes))]

    order = np.concatenate(label_list)
    resorted = dmat[:, order][order]
    
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    plt.imshow(resorted, cmap='inferno')

    for s, l in zip(start_points, cluster_sizes):
        rect = patches.Rectangle((s-.5, s-0.5), l, l, linewidth=3,
                                 edgecolor='w', facecolor='none')
        ax.add_patch(rect)

    plt.yticks([])
    plt.xticks([])

    plt.show()


def single_study_eval(df_subs: pd.DataFrame, df_dist: pd.DataFrame) -> None:
    # Do clustering, plotting, & similarity computation for each dataset

    combinations = df_subs.value_counts(['dataset', 'atlas']).index

    # Define convenience slicing lambda
    slice_da = lambda _df, dset, atlas: _df[(_df['dataset'] == dset) &
                                            (_df['atlas'] == atlas)]

    clustering = []
    for ds, at in combinations:
        t_subs = slice_da(df_subs, ds, at)
        t_dist = slice_da(df_dist, ds, at)['distance'].values[0]

        t_pt = "{0} ({1})".format(ds, at)
        t_labels, t_clust = cluster_subjects(t_subs, t_dist, plot_title=t_pt)

        clustering += [{
            "dataset": ds,
            "atlas": at,
            "labels": t_labels,
            "clustering": t_clust
        }]
        
        label_list = [_['members'] for _ in t_clust]
        print(ds, at, np.mean(t_dist), np.std(t_dist))
        plot_clustered_dset(t_dist, label_list)


def main():
    parser = ArgumentParser()
    parser.add_argument("dset", help="Path to dataset director(y/ies)")

    results = parser.parse_args()

    dset = Path(results.dset)

    # Load dataframes and distance matrices
    df_subs = load_and_stack(dset, pattern="*similarity*h5")
    dists = [{ 'dataset': _.name.split('_')[0],
               'atlas':_.name.split('_')[1],
               'distance': np.loadtxt(_) }
             for _ in dset.rglob('*distmat*txt')]
    df_dist = pd.DataFrame.from_dict(dists)

    single_study_eval(df_subs, df_dist)


if __name__ == "__main__":
    main()

