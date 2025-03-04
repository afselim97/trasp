#%%
import numpy as np
import itertools
from tqdm import tqdm
from typing import List
from numpy.typing import NDArray
import glob
import os
import tskit

def fraction_shared_nodes(tree1,tree2,samples):
    nodes1 = np.array(list(tree1.nodes()))
    nodes1 = nodes1[~np.isin(nodes1,samples)]
    nodes2 = np.array(list(tree2.nodes()))
    nodes2 = nodes2[~np.isin(nodes2,samples)]

    shared_nodes = np.intersect1d(nodes1,nodes2)
    return len(shared_nodes)/len(nodes1)

def compute_distance_till_threshold(ts,threshold=0.05):
    samples = ts.samples()
    n=len(samples)
    trees = ts.trees()
    first_tree = ts.first(sample_lists=True)
    if len(list(first_tree.nodes())) == len(samples):
        first_tree = ts.at_index(1)
        next(trees)
    i=0
    for i,tree in enumerate(trees):
        x = fraction_shared_nodes(first_tree,tree,samples)
        if x<threshold:
            if i<n/4: # That is if for some reason the threshold is reached too quickly. We set the distance till threshold manually to every n trees
                return n
            else:
                return i
    return n

def fill_matrix(matrix: NDArray[np.uint16], leaf_sets: List[List[int]], time_inxs: int) -> NDArray:
    num_sets = len(leaf_sets)
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            samples_1 = leaf_sets[i]
            samples_2 = leaf_sets[j]
            matrix[np.ix_(samples_1, samples_2, time_inxs)] += 1
            matrix[np.ix_(samples_2, samples_1, time_inxs)] += 1
    return matrix

#%%
class trasp():
    """
    An object used to calculate pairwise coalescent rates as a function of time
    """
    def __init__(
        self,
        ts: tskit.TreeSequence
        ) -> None:
        """
        Initializes the trasp object
        Args:
            tree_sequences (tskit.TreeSequence): A tree sequence object
        """
        # Basic definitions
        self.samples = ts.samples()
        self.n = len(self.samples)
        self.node_times = ts.nodes_time.copy()
        distance_till_threshold = compute_distance_till_threshold(ts)
        self.L = int(ts.num_trees/distance_till_threshold)+1

        self.trees = itertools.islice(ts.trees(), 0, None, distance_till_threshold)

    def compute_coal_counts(self,t_list_lower: List[float], t_list_upper: List[float], window_list_upper: List[float]) -> NDArray:
        """Computes time-varying coalescent rates between each two pairs of samples using a Nelson Aalen Estimate

        Args:
            t_list (List[float]): The list of time points that define the top bound of each time window. The first time point will always start from zero.
            delta_list (List[float]): A list of window sizes

        Returns:
            NDArray[int]: Number of uncoalesced trees between each two samples as a function of time
            NDArray[int]: Number of trees where a coalescence event occured between each two samples within each time window
            NDArray[int]: Pairwise coalescent rate as a function of time.
        """
        num_coal_events_between_windows = np.zeros((self.n,self.n,len(t_list_lower)),np.uint16)
        num_coal_events_within_windows = np.zeros((self.n,self.n,len(t_list_lower)),np.uint16)

        for tree in self.trees:
            nodes = np.array(list(tree.nodes()))
            nodes = nodes[~np.isin(nodes,self.samples)]
            for node in nodes:
                leaf_sets = [list(tree.leaves(child)) for child in tree.children(node)]
                node_time = self.node_times[node]
                above_lower_bound = node_time >= t_list_lower
                below_coal_upper_bound = node_time < t_list_upper 
                below_window_upper_bound = node_time < window_list_upper
                coal_indices = np.where(np.logical_and(above_lower_bound,below_coal_upper_bound))[0]
                coalesced_within_windows_indices = np.where(np.logical_and(above_lower_bound,below_window_upper_bound))[0]

                if len(coal_indices) > 0:
                    num_coal_events_between_windows = fill_matrix(num_coal_events_between_windows,leaf_sets,coal_indices)
                if len(coalesced_within_windows_indices) > 0:
                    num_coal_events_within_windows = fill_matrix(num_coal_events_within_windows,leaf_sets,coalesced_within_windows_indices)

        num_coal_events_between_windows[np.arange(self.n),np.arange(self.n),:] = 0
        return num_coal_events_between_windows, num_coal_events_within_windows


def calculate_rates(ts_dir,t_list,delta):
    delta_list = np.array([delta]*len(t_list))
    if t_list[0] == 0:
        t_list_lower = t_list
        window_list_upper = t_list_lower + delta_list
        t_list_upper = np.concatenate((t_list[1:],[window_list_upper[-1]]))
    else:
        t_list_lower = np.concatenate(([0],t_list))
        delta_list = np.concatenate(([delta_list[0]],delta_list))
        window_list_upper = t_list_lower + delta_list
        t_list_upper = np.concatenate((t_list,[window_list_upper[-1]]))
    
    L_list = []
    num_coal_events_between_windows_total = None
    num_coal_events_within_windows_total = None
    ts_files = glob.glob(os.path.join(ts_dir,"*.trees"))
    for file in tqdm(ts_files):
        ts = tskit.load(file)
        trasp_object = trasp(ts)
        L_list.append(trasp_object.L)
        num_coal_events_between_windows,num_coal_events_within_windows = trasp_object.compute_coal_counts(t_list_lower,t_list_upper,window_list_upper)
        if num_coal_events_between_windows_total is None:
            num_coal_events_between_windows_total = num_coal_events_between_windows
            num_coal_events_within_windows_total = num_coal_events_within_windows
        else:
            num_coal_events_between_windows_total += num_coal_events_between_windows
            num_coal_events_within_windows_total += num_coal_events_within_windows
    n = num_coal_events_between_windows_total.shape[0]
    L = sum(L_list)
    num_coalesced_through_time_total = np.cumsum(num_coal_events_between_windows,axis=2)
    num_coalesced_through_time_total = np.dstack((np.zeros((n,n),np.uint16),num_coalesced_through_time_total)) # Adding the zero time points
    num_uncoalesced_through_time_total = L - num_coalesced_through_time_total
    num_uncoalesced_through_time_total = num_uncoalesced_through_time_total[:,:,:-1]

    rates_through_time = num_coal_events_within_windows_total / (delta_list*num_uncoalesced_through_time_total+1e-9) ## Nelson Aalen estimate of pairwise coalescent rates
    rates_through_time[np.arange(n),np.arange(n),:] = 0
    num_uncoalesced_through_time_total[np.arange(n),np.arange(n),:] = 0

    if t_list[0] !=0:
        num_uncoalesced_through_time_total = num_uncoalesced_through_time_total[:,:,1:]
        num_coal_events_within_windows_total = num_coal_events_within_windows_total[:,:,1:]
        rates_through_time = rates_through_time[:,:,1:]

    return num_uncoalesced_through_time_total,num_coal_events_within_windows_total,rates_through_time
# %%