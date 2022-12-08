#!/usr/bin/env python

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import json


from sklearn.manifold import TSNE, SpectralEmbedding, MDS
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import rankdata, kendalltau


with open('./cmap/cmap.json', 'r') as fhandle:
    colors = json.load(fhandle)

def colour(row: pd.Series) -> str:
    # Identify color
    ids = row['alignment'], row['filtering'], row['scrubbing'], row['gsr']
    key = ['0' if i == "ANTs" or i == True else '1' for i in ids]
    key = "[" + " ".join(key) + "]"
    return colors[key]


def annotate(row: pd.Series) -> str:
    # Create annotations
    ids = row['subject'], row['session'], row['alignment'], row['filtering'], row['scrubbing'], row['gsr']
    anno = "Sub: {0}<br>Ses: {1}<br>Alignment: {2}<br>Filter: {3}<br>Scrub: {4}<br>GSR: {5}".format(*ids)
    return anno


def load_and_prepare(path: Path, atlas: str='cc2') -> pd.DataFrame:
    files = [_ for _ in path.rglob('*{0}*'.format(atlas))]
    df = pq.ParquetDataset(files)
    pddf = df.read().to_pandas()

    pddf['color'] = pddf.apply(colour, axis=1)
    pddf['annotation'] = pddf.apply(annotate, axis=1)
    return pddf


def embed_data(df: pd.DataFrame, outdir: Path, dset: str,
                               nc: int=3, rs: int=42) -> None:
    X = np.stack(df['graph'].values).astype(np.float64)
    cs = df['color'].values
    annos = df['annotation'].values

    strats = [TSNE(n_components=nc, random_state=rs,
                   init='pca', perplexity=32),
              MDS(n_components=nc, random_state=rs),
              SpectralEmbedding(n_components=nc, random_state=rs)]

    gsm = go.scatter.Marker
    for strat in strats:
        x_ld = strat.fit_transform(X)

        fig = make_subplots(rows=x_ld.shape[1]-1, cols=x_ld.shape[1]-1,
                            subplot_titles=['Dimension {0}'.format(_+2)
                                            for _ in range(x_ld.shape[1]-1)])

        for dim1, dim2 in combinations(range(x_ld.shape[1]), r=2):
            xs = list(x_ld[:, dim2])
            ys = list(x_ld[:, dim1])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                                     marker=gsm(size=12, opacity=0.5),
                                     marker_color=cs, hovertext=annos,
                                     showlegend=False), row=dim1+1, col=dim2)

            if dim2 - dim1 == 1:
                fig.update_yaxes(title_text="Dimension {0}".format(dim1+1),
                                 row=dim1+1, col=dim2)

        ename = str(type(strat)).split('.')[-1].strip("'>")
        fig.update_layout(margin=dict(l=20, r=20, t=80, b=20), title=ename)
        fig.write_html(outdir / "{0}_{1}_embedding.html".format(dset, ename))
        fig.write_image(outdir / "{0}_{1}_embedding.pdf".format(dset, ename))


def compute_similarity(df: pd.DataFrame, outdir: Path, dset: str) -> \
                                                      (pd.DataFrame, str):
    # Define columns of interest (COIs) and get unique subject x session combos
    cois = ['gsr', 'alignment', 'filtering', 'scrubbing']
    subses = df[['subject', 'session']].value_counts().index
    # N.B.: I put GSR first because it is likely to drive clustering, so,
    #  without re-sorting the columns, we'll still have a block-structure

    # Create empty list to store similarity results
    similarity = []
    for idx, ss in enumerate(subses):
        # Filter df by subject x session, and sort by COIs
        tmpdf = df.query("subject == '{0}' and session == '{1}'".format(*ss))
        tmpdf = tmpdf.sort_values(cois).reset_index()

        # Used to ensure sorting is always consistent
        ids = str(tmpdf[cois].values)
        if idx == 0:
            ref_order = ids
        assert(ids == ref_order)

        # Grab graphs, store in obs x edges matrix, and compute correlations
        graphs = np.stack(tmpdf['graph'].values)
        corr = np.corrcoef(graphs)

        # Rank the correlations, and then normalize
        N = len(tmpdf)
        rank_corr = np.reshape(rankdata(corr), (N, N)) * 1.0/ corr.size

        # Create entry for (eventual) dataframe
        similarity += [{
            'subject': ss[0],
            'session': ss[1],
            'corr': corr,
            'rank_corr' : rank_corr
        }]

        # Optionally, plot
        # TODO: add writing to file here
        # sns.clustermap(rank_corr)

    return (pd.DataFrame.from_dict(similarity), ref_order)


def subject_comparison(df: pd.DataFrame, outdir: Path, dset: str) -> \
                                                (pd.DataFrame, np.array):
    """
    Inputs:
      df: pd.DataFrame
            Should contain ranked similarity matrices. Expected columns:
            [subject, session, corr, rank_corr]
      outdir: Path
              Location for saving output data and plots
    
    Returns:
      pd.DataFrame
        Will contain pairwise similarity scores across all rows in df
      np.array
        Will contain an N x N matrix of similarity scores, where N is the
        number of rows in df
    """
    # Create empty list and matrix to store the consistency results
    consistency = []
    N = len(df)
    dist = np.zeros((N, N))

    # Create lambda for inserting comparisons in the matrix
    sub = sorted(df['subject'].unique())
    ses = sorted(df['session'].unique())
    nses = len(ses)
    loc = lambda su, se: sub.index(su)*nses + ses.index(se)

    # Compare all pairs of subjects x sessions
    for idx, r1 in df.iterrows():
        for jdx, r2 in df.iloc[idx:].iterrows():
            loc1 = loc(r1['subject'], r1['session'])
            loc2 = loc(r2['subject'], r2['session'])
    
            # Compute Kendall Tau statistic (aka bubble-sort distance)
            csim = kendalltau(r1['rank_corr'], r2['rank_corr'])[0]
            dist[loc1, loc2] = csim

            # Prepare dataframe entry
            consistency += [{
                'subject1': r1['subject'],
                'session1': r1['session'],
                'subject2': r2['subject'],
                'session2': r2['session'],
                'consistency': csim
            }]

    # Complete the matrix
    dist += dist.T - np.eye(N)

    # Optionally, plot
    sns.clustermap(dist)
    plt.savefig(outdir / "{0}_consistency.png".format(dset))

    return (pd.DataFrame.from_dict(consistency), dist)


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", help="Link to a directory of organized"
                        " parquet datasets.")
    parser.add_argument("output_path", help="Directory for writing outputs")
    parser.add_argument("--atlas", choices=["des", "hox", "cc2", "aal"],
                        default="des", help="Parcellation ID to analyze")
    parser.add_argument("--embed", action="store_true", help="Flag determining"
                        " whether or not to produce tSNE, MSD, and SE plots")

    # Parse inputs and prepare useful variables
    results = parser.parse_args()
    path = Path(results.dataset_path)
    outdir = Path(results.output_path) / path.name
    outdir.mkdir(parents=True, exist_ok=True)
    modif = str(path.name) + "_" + results.atlas

    # Grab data
    df = load_and_prepare(path, atlas=results.atlas)

    # Compute pipeline similarities, and then consolidate results
    outdir_s = outdir / "similarities"
    outdir_s.mkdir(exist_ok=True)
    df_sim, sorting = compute_similarity(df, outdir_s, dset=modif)
    df_con, dist = subject_comparison(df_sim, outdir_s, dset=modif)

    # Embed all the data using a few different approaches
    if results.embed:
        outdir_e = outdir / "embedding"
        outdir_e.mkdir(exist_ok=True)
        embed_data(df, outdir_e, dset=modif)


if __name__ == "__main__":
    main()

