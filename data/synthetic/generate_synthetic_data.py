# %%
# %pip install jax==0.3.15 jax-md seaborn
# %%
import numpy as np
import pandas as pd
import numpy as onp
import os
from os import path
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
# %%
from jax.config import config ; config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as np
from jax import random
from jax import lax
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
 
sns.set_style(style='white')

def format_plot(x, y):  
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)
  
def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

# %%
N = 400
dimension = 2
box_size = quantity.box_size_at_number_density(N, 1, dimension)
dt = 1e-3
displacement, shift = space.periodic(box_size) 
kT = 0.1

# %%
key = random.PRNGKey(0)
key, split = random.split(key)

initial_positions = box_size * random.uniform(split, (N, dimension), dtype=np.float64)
species = np.array([0]*(N//2) + [1]*(N-N//2))
sigmas = np.array([[1.0, 1.2], [1.2, 1.4]])

energy_fn = energy.soft_sphere_pair(displacement, sigma=sigmas, species=species)
init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, kT)
state = init_fn(key, initial_positions)
# %%
# %%
write_count = 100
write_every = 10

def simulate_brownian(write_every):
  def step_fn(i, state_log):
    state, log = state_log
    log['position'] = lax.cond(i % write_every == 0,
                               lambda p: \
                               p.at[i // write_every].set(state.position),
                               lambda p: p,
                               log['position'])
    state = apply_fn(state, kT=kT)
    return state, log
  os.makedirs(f"write_every_{write_every}", exist_ok=True)
  steps = write_every*write_count
  for j in range(10):
    log = {
      'position': np.zeros((steps // write_every,) + initial_positions.shape) 
    }
    _, log = lax.fori_loop(0, steps, step_fn, (state, log))
    np.save(f"write_every_{write_every}/{j}.npy",np.array(log["position"]))

    fig = plt.figure()
    ims = []
    for t in range(len(log['position'])):
      im = plt.plot(log["position"][t, :, 0], log["position"][t, :, 1], ".b")
      ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(f"write_every_{write_every}/{j}.gif", fps=4)

for write_every in [25,50,75,100,125,150,175,200]:
  print(write_every)
  simulate_brownian(write_every)

# %%

def coordinate_to_synthetic_data(coordinate):
  dfs = []
  for track_id in range(coordinate.shape[1]):
    c=coordinate[:, track_id]
    c_prev=c[0]
    subtrack_id = 0
    subtrack_ids = [f"{track_id}-0"]
    for _c in c[1:]:
      if np.linalg.norm(_c-c_prev)>10:
        subtrack_id += 1
      subtrack_ids.append(f"{track_id}-{subtrack_id}")
      c_prev=_c

    df=pd.DataFrame({
      "t": np.arange(len(c)),
      'y': c[:, 0],
      'x': c[:, 1],
      "track_id": subtrack_ids
    })
    dfs.append(df)
  df = pd.concat(dfs)
  unique_id=onp.array(df["track_id"].unique())
  track_id_map=dict(zip(unique_id,onp.arange(len(unique_id))))
  df["track_id2"] = df["track_id"].map(track_id_map)
  return df

# %%
test_coordinates = np.load("write_every_25/0.npy")
df=coordinate_to_synthetic_data(test_coordinates)

ims = []
colors = {j:plt.cm.tab20(onp.random.rand()) for j in df["track_id2"].unique()}
for t, grp in df.groupby("t"):
  im = plt.scatter(grp["y"], grp["x"], c=[colors[i] for i in grp["track_id2"]])
  plt.show()
  if t>20:
    break

# %% 
# assign random fluorescent labels to each track
import ray
ray.init(num_cpus=10)

@ray.remote
def convert_file(file_name):
  csv_name=file_name.replace(".npy",".csv")
  if path.exists(csv_name):
    return
  coordinates=np.load(file_name)
  df=coordinate_to_synthetic_data(coordinates)
  df.to_csv(csv_name,index=False)

files = glob("write_every_*/*.npy")
res = ray.get([convert_file.remote(f) for f in files])

# %%
n_colors = 8
def assign_random_fluorescent_labels(df):
  track_ids = df["track_id2"].unique()
  to_color = dict(zip(track_ids, onp.random.choice(n_colors, len(track_ids))))
  df["color"] = df["track_id2"].map(to_color)
  for i in range(3):
    has_digit = list(map(lambda x:f'{x>>i:b}'[-1]=='1', df["color"].values))
    df[f"channel_{i}"] = onp.clip(
      onp.where(has_digit, 
      onp.random.normal(6,0.5,len(df)), 
      onp.random.normal(2,0.5,len(df)), 
    ),0, 10)
  return df

df = pd.read_csv("write_every_25/0.csv")
df = assign_random_fluorescent_labels(df)
df
# %%
df.rename(columns={"t":"frame","track_id2":"track"}).drop(columns=["track_id"])

# %%
organized_data_path = "organized_data"
os.makedirs(organized_data_path, exist_ok=True)

def convert_file2(file_name):
  output_path = path.join(organized_data_path, 
                          path.dirname(file_name), 
                          path.basename(file_name).replace(".csv",""))
  if path.exists(output_path):
    return
  print(file_name)
  df = pd.read_csv(file_name)
  df = assign_random_fluorescent_labels(df)
  df2=df.rename(columns={"t":"frame","track_id2":"track"}).drop(columns=["track_id"])
  new_file_name = path.join(output_path,"regionprops.csv")
  os.makedirs(path.dirname(new_file_name), exist_ok=True)
  df2.to_csv(new_file_name)
  track_txt_path=path.join(output_path,"02_GT/TRA/man_track.txt")
  tracking_txt_df=[]
  for track, grp in df2.groupby("track"):
    tracking_txt_df.append(
        [
            track,
            grp["frame"].min(),
            grp["frame"].max(),
            0
        ]
    )
  os.makedirs(path.dirname(track_txt_path), exist_ok=True)
  onp.savetxt(track_txt_path,np.array(tracking_txt_df,dtype=np.uint32),fmt="%d")

files = glob("write_every_*/*.csv")
for f in tqdm(files):
  convert_file2(f)



# %%
organized_data_path = "organized_data_small"
os.makedirs(organized_data_path, exist_ok=True)

def convert_file2(file_name):
  output_path = path.join(organized_data_path, 
                          path.dirname(file_name), 
                          path.basename(file_name).replace(".csv",""))
  if path.exists(output_path):
    return
  print(output_path)
  df = pd.read_csv(file_name)
  df = assign_random_fluorescent_labels(df)
  df2=df.rename(columns={"t":"frame","track_id2":"track"}).drop(columns=["track_id"])
  df2 = df2[df2["frame"]<20]
  new_file_name = path.join(output_path,"regionprops.csv")
  os.makedirs(path.dirname(new_file_name), exist_ok=True)
  df2.to_csv(new_file_name)
  track_txt_path=path.join(output_path,"02_GT/TRA/man_track.txt")
  tracking_txt_df=[]
  for track, grp in df2.groupby("track"):
    tracking_txt_df.append(
        [
            track,
            grp["frame"].min(),
            grp["frame"].max(),
            0
        ]
    )
  os.makedirs(path.dirname(track_txt_path), exist_ok=True)
  onp.savetxt(track_txt_path,np.array(tracking_txt_df,dtype=np.uint32),fmt="%d")

files = glob("write_every_*/*.csv")
for f in tqdm(files):
  convert_file2(f)
# %%
