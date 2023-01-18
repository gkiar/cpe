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
import os


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


def compute_similarity(df: pd.DataFrame, outdir: Path, dset: str,
                       plot: bool = False) -> (pd.DataFrame, str):
    # Define columns of interest (COIs) and get unique subject x session combos
    cois = ['gsr', 'alignment', 'filtering', 'scrubbing']
    subses = df[['subject', 'session']].value_counts().index
    # N.B.: I put GSR first because it is likely to drive clustering, so,
    #  without re-sorting the columns, we'll still have a block-structure

    # Create empty list to store similarity results
    outpattr = str(outdir / "individual") + "/{0}_{1}_consistency.png"
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

        if plot:
            # Plot
            plt.imshow(rank_corr)
            plt.title("{0} {1}".format(dset, ss))
            plt.savefig(outpattr.format(dset, ss))

    if plot:
        os.system("gifski -r 6 -o {0} {1}".format(outdir/ "{0}_individual.gif".format(dset),
                                                  outpattr.format(dset, '*')))

    return (pd.DataFrame.from_dict(similarity), ref_order)


def subject_comparison(df: pd.DataFrame) -> (np.array):
    """
    Inputs:
      df: pd.DataFrame
            Should contain ranked similarity matrices. Expected columns:
            [subject, session, corr, rank_corr]
    
    Returns:
      np.array
        Will contain an N x N matrix of similarity scores, where N is the
        number of rows in df
    """
    # Create empty list and matrix to store the consistency results
    consistency = []
    N = len(df)
    dist = np.zeros((N, N))

    # Sort values to ensure neighbours end up next to one another in the plot
    df = df.sort_values(['subject', 'session'])

    # Compare all pairs of subjects x sessions
    for idx, r1 in df.iterrows():
        for jdx, r2 in df.iterrows():
            if jdx < idx:
                continue

            # Compute Kendall Tau statistic (aka bubble-sort distance)
            csim = kendalltau(r1['rank_corr'], r2['rank_corr'])[0]
            dist[idx, jdx] = csim

    # Complete the matrix
    dist += dist.T - np.eye(N)
    return dist


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", help="Link to a directory of organized"
                        " parquet datasets.")
    parser.add_argument("output_path", help="Directory for writing outputs")
    parser.add_argument("--atlas", choices=["des", "hox", "cc2", "aal"],
                        default="des", help="Parcellation ID to analyze")
    parser.add_argument("--plot", action="store_true", help="Use if you want "
                        "to visualize data")
    parser.add_argument("--embed", action="store_true", help="Flag determining"
                        " whether or not to produce tSNE, MSD, and SE plots")

    # Parse inputs and prepare useful variables
    results = parser.parse_args()
    path = Path(results.dataset_path)
    outdir = Path(results.output_path) / path.name
    outdir.mkdir(parents=True, exist_ok=True)
    modif = str(path.name) + "_" + results.atlas
    pl = results.plot

    # Grab data
    df = load_and_prepare(path, atlas=results.atlas)

    # Compute pipeline similarities, and then consolidate results
    outdir_s = outdir
    if pl:
        (outdir_s / "individual").mkdir(parents=True, exist_ok=True)
    df_sim, sorting = compute_similarity(df, outdir_s, dset=modif, plot=pl)
    dist = subject_comparison(df_sim) 

    # Write out results
    basename = str(outdir_s / modif)
    df_sim.to_hdf(basename + "_similarity.h5", 'dset')
    with open(basename + "_sorting.txt", "w") as fhandle:
        fhandle.write(sorting)
    with open(basename + "_distmat.txt", "w") as fhandle:
        np.savetxt(fhandle, dist)

    # Embed all the data using a few different approaches
    if results.embed:
        outdir_e = outdir / "embedding"
        outdir_e.mkdir(parents=True, exist_ok=True)
        embed_data(df, outdir_e, dset=modif)


if __name__ == "__main__":
    main()

