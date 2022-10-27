import numpy as np
import networkx as nx
import os
import pandas as pd
from os import path
from laptrack.utils import order_edges
def save_evaluation_platform_input(coords,predicted_edges,output_dir,trial_str):
    ## ignore the splitting as the original score does not include division
    pred_tree = nx.from_edgelist(
            order_edges(predicted_edges), create_using=nx.DiGraph
    )

    ### output result for evaluation by evaluation platform (yeast image toolkit) ###
    # 
    #Frame_number, Cell_number, Position_X, Position_Y
    seg_res = []
    for frame, cs in enumerate(coords):
        for ind,c in enumerate(cs):
            seg_res.append([frame+1,ind+1,c[0],c[1]])
    seg_df = pd.DataFrame(np.array(seg_res),columns=["Frame_number", "Cell_number", "Position_X", "Position_Y"])
    seg_df["Frame_number"] = seg_df["Frame_number"].astype(int)
    seg_df["Cell_number"] = seg_df["Cell_number"].astype(int)
    seg_df.to_csv(path.join(output_dir,f"res_seg_{trial_str}.txt"),index=False)
    dividing_edges = set(pred_tree.edges) - set(predicted_edges)
    for e in dividing_edges:
        pred_tree.remove_edge(*e)
    tra_res = []
    # Frame_number, Cell_number, Unique_cell_number
    for j,segment in enumerate(nx.connected_components(pred_tree.to_undirected())):
        for frame, ind in segment:
            tra_res.append([frame+1,ind+1,j+1])
    tra_df = pd.DataFrame(np.array(tra_res),columns=["Frame_number", "Cell_number", "Unique_cell_number"],dtype=int).sort_values(["Frame_number", "Cell_number"])
    tra_df.to_csv(path.join(output_dir,f"res_tra_{trial_str}.txt"),index=False)
    #
    ################################################################################## 