import numpy as np
from numpy import random
from scipy import linalg as la
from scipy import spatial as spt
from sklearn.neighbors import NearestNeighbors as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

def tally_folders():
    os.chdir("C:/Dev/Clustering/Wildlife Acoustics/Kaggle/8")
    df = pd.read_csv("cluster.csv")
    list(df.columns)
    grouped = df.groupby(['FOLDER', 'TOP1MATCH*'], as_index=False)
    # delineate each species-cluster assignment
    g1 = grouped['DURATION'].sum()
    # count each cluster
    grouped2 = df.groupby(['TOP1MATCH*'], as_index=False)
    g2 = grouped2['DURATION'].sum()
    gm = g1.merge(g2, how='inner', on="TOP1MATCH*", suffixes=("_species-cluster", "_cluster_total"),)


    grouped3 = df.groupby(['FOLDER'], as_index=False)
    g3 = grouped3['DURATION'].sum()
    gm = gm.merge(g3, how='inner', on="FOLDER")
    gm.rename(columns={'DURATION': 'DURATION_species_total'}, inplace=True)
    # extent to which a single species dominates a cluster -- ie, separation of cluster from other species
    gm['Separation'] = gm['DURATION_species-cluster']/gm['DURATION_cluster_total']
    # extent to which a species is dominated by a cluster
    gm['Coverage'] = gm['DURATION_species-cluster']/gm['DURATION_species_total']
    gm['Characteristic'] = 2*gm['Separation']*gm['Coverage']/(gm['Separation'] + gm['Coverage'])
    gm['Characteristic'] > 0.1
    #gm = gm.sort_values('characteristic', ascending=False)
    final = gm.loc[gm['Characteristic']>0.1]
    analysis = final.groupby(['FOLDER']).size()
    print("Species with at least one characteristic cluster: ", len(analysis))
    print("Total number of characteristic clusters: ", sum(analysis))
    #assignments = test2[test2 > 50]

if __name__ == '__main__':
    random.seed()
    #travis_prototypes()
    tally_folders()
    print("Done")
