# mypy: ignore-errors
from tqdm import tqdm
from sklearn.manifold import smacof
from utils import matrix_to_df
from numpy.typing import NDArray
import polars as pl
import warnings
warnings.filterwarnings("ignore")
#%%
# def rates_to_distances(rates_over_time):
#     n = rates_over_time.shape[0]
#     rates_over_time[rates_over_time==0] = np.min(rates_over_time[rates_over_time!=0])*0.5
#     D = -np.log(rates_over_time/np.max(rates_over_time))
#     D[np.arange(n),np.arange(n),:] = 0

#     return D
#%%
# def df_to_matrix(df):
#     r = df["row"].max() + 1
#     c = df["column"].max() + 1
#     d = df["depth"].max() + 1

#     element_cols = [col for col in df.columns if col.startswith("element")]
#     p = len(element_cols)
#     elements = df[element_cols].to_numpy()

#     matrix = elements.reshape(r,c,d,p)

#     return matrix
# data = pd.read_parquet("data/results/Zigzag_1S14_inferred/data.parquet")
# data = data.rename(columns={"sample_1_inx": "row", "sample_2_inx": "column", "time_inx": "depth","num_uncoalesced": "element_1","num_coal_events": "element_2","raw_rate": "element_3"})
# x = df_to_matrix(data)
# rates = x[:,:,:,-1]
# times = np.unique(data["time"])
#%%
def rates_to_distances(rates_over_time):
    upper_bound = np.quantile(rates_over_time,0.99)
    lower_bound = np.quantile(rates_over_time,0.01)
    if lower_bound == 0:
        lower_bound = np.min(rates_over_time[rates_over_time!=0])
    rates_over_time[rates_over_time>upper_bound] = upper_bound
    rates_over_time[rates_over_time<lower_bound] = lower_bound

    n = rates_over_time.shape[0]
    # rates_over_time[rates_over_time==0] = np.min(rates_over_time[rates_over_time!=0])*0.5
    D = -np.log(rates_over_time/(np.max(rates_over_time)))
    # D = 1-(100*rates_over_time)
    # a = np.max(D)
    # b = np.min(D)
    # D = (D-b)/(a-b)
    
    D[np.arange(n),np.arange(n),:] = 0
    return D
# D = rates_to_distances(rates)
# mean_D = [np.mean(D[:,:,t][D[:,:,t]!=0]) for t in range(len(times))]
# plt.scatter(times[50:],mean_D[50:])
# plt.xscale("log")
# plt.ylim(0,1.5)
#%%
def compute_embeddings_over_time(rates_over_time,n_components: int = 2):
    """Computes PCs over time given a matrix over time

    Args:
        rates_over_time (NDArray): different matrices over the third dimension is over different time points
        n_components (int, optional): num components to compute PCA. Defaults to 2.
    """

    n = rates_over_time.shape[0]
    n_timepoints = rates_over_time.shape[-1]
    X_over_time = np.zeros((n,n_components,n_timepoints))
    D = rates_to_distances(rates_over_time)

    for k in tqdm(range(n_timepoints)):
        inx = n_timepoints-k-1
        if k==0:
            X, _ = smacof(D[:,:,inx], n_components=n_components)
        else:
            X, _ = smacof(D[:,:,inx], n_components=n_components,init=init)
        X_over_time[:,:,inx] = X
        init = X
    return X_over_time

def create_embeddings_df(matrix: NDArray, times: NDArray, samples_dict: dict) -> pl.DataFrame:
    # Convert matrix to Polars DataFrame
    df = matrix_to_df(matrix)

    # Rename columns
    df = df.rename({
        "row": "sample_id",
        "column": "component",
        "depth": "time_inx",
        "element_1": "embedding"
    })

    # Calculate delta (difference between times)

    # Create dictionaries for mappings
    sample_to_pop_dict = {sample: pop for pop, samples in samples_dict.items() for sample in samples}
    times_dict = {i: t for i, t in enumerate(times)}

    # Map values to new columns
    df = df.with_columns([
        pl.col("sample_id").map_elements(lambda x: sample_to_pop_dict.get(x, "unknown"),return_dtype=str).alias("population"),
        pl.col("time_inx").map_elements(lambda x: times_dict.get(x, np.nan),return_dtype=float).alias("time"),
        (pl.col("component") + 1).alias("component")
    ])
    return df

import numpy as np

def procrustes_adjustment(X, Y):
    
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    
    cov_matrix = np.dot(Y_centered.T, X_centered)
    
    
    U, _, Vt = np.linalg.svd(cov_matrix)
    rotation_matrix = np.dot(U, Vt)
    
    
    Y_rotated = np.dot(Y_centered, rotation_matrix)
    
    return Y_rotated

