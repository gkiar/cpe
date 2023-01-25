### Steps to reproduce this analysis

1. Obtain data from [this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009279) in `graphml` format.
2. Run the `data_organize.py` script to tabulate and organize the graphs into a series of `parquet` tables.
3. Run the `compute_similarity.py` script to calculate the similarities across pipeline pairs for all datasets.
4. Run the `cluster.py` script to cluster the pipeline similarity matrices.
