# %%
from utils.data_loader import read_data
from utils.common import read_yaml
# %%
rpk = read_yaml("../setting_yaml/homeostasis.yaml")["regionprop_keys"]
coords1, track_labels, true_edges, GT_TRA_images = read_data("../data/homeostasis/organized_data/area1", rpk)
coords2, track_labels, true_edges, GT_TRA_images = read_data("../data/homeostasis/organized_data/area2", rpk)
print(min([c.shape[0] for c in coords1]), max([c.shape[0] for c in coords1]))
print(min([c.shape[0] for c in coords2]), max([c.shape[0] for c in coords2]))
# %%
rpk = read_yaml("../setting_yaml/CellMigration.yaml")["regionprop_keys"]
coords1, track_labels, true_edges, GT_TRA_images = read_data("../data/CellMigration/organized_data/Sparse1", rpk)
print(min([c.shape[0] for c in coords1]), max([c.shape[0] for c in coords1]))
#
# %%
