import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx

from typing import List, Tuple, Optional

def create_QM9_train_val_test_nx(train_ratio: float = 0.8,
                               val_ratio: float = 0.1,
                               test_ratio: float = 0.1,
                               subset_size: Optional[int] = None,
                               random_seed: int = 42) -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph], List[Data], List[Data], List[Data]]:
    """
    Load in the QM9 dataset and splits it into training, validation, and test sets
    of NetworkX graphs and their corresponding PyG Data objects.

    Args:
        train_ratio: Proportion of the dataset to allocate for training.
        val_ratio: Proportion of the dataset to allocate for validation.
        test_ratio: Proportion of the dataset to allocate for testing.
        subset_size: Optional. If provided, a random subset of this size will be used from the original dataset.
        random_seed: Seed for shuffling to ensure reproducibility.

    Returns:
        A tuple containing:
            (train_nx, val_nx, test_nx, train_pyg, val_pyg, test_pyg).
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Sum of ratios must be equal to 1.")

    np.random.seed(random_seed)

    dataset = QM9(root='./data/QM9') 

    if subset_size is not None and subset_size < len(dataset):
        # Get a permuted list of indices from the original dataset
        indices_full_dataset = np.random.permutation(len(dataset))[:subset_size]
        # Create the current_dataset by slicing the original dataset
        current_dataset = dataset[indices_full_dataset.tolist()]
    else:
        current_dataset = dataset

    num_graphs = len(current_dataset)
    # These indices are local to current_dataset
    permuted_local_indices = np.random.permutation(num_graphs)

    train_end = int(train_ratio * num_graphs)
    val_end = train_end + int(val_ratio * num_graphs)

    train_local_idx = permuted_local_indices[:train_end]
    val_local_idx = permuted_local_indices[train_end:val_end]
    test_local_idx = permuted_local_indices[val_end:]

    def convert_and_collect(local_indices_list):
        nx_graphs_list = []
        pyg_graphs_list = []
        for local_idx in local_indices_list:
            # Access items from current_dataset using the local permuted indices
            pyg_graph = current_dataset[int(local_idx)]
            pyg_graphs_list.append(pyg_graph)

            g = to_networkx(pyg_graph, node_attrs=['x'], to_undirected=True)
            nx_graphs_list.append(g)
        return nx_graphs_list, pyg_graphs_list

    train_nx, train_pyg = convert_and_collect(train_local_idx)
    val_nx, val_pyg = convert_and_collect(val_local_idx)
    test_nx, test_pyg = convert_and_collect(test_local_idx)

    print(f'Original dataset size: {len(dataset)}')
    if subset_size is not None:
        print(f'Using subset of size: {len(current_dataset)}')
    print(f'Train NX graphs: {len(train_nx)}, Train PyG graphs: {len(train_pyg)}')
    print(f'Validation NX graphs: {len(val_nx)}, Validation PyG graphs: {len(val_pyg)}')
    print(f'Test NX graphs: {len(test_nx)}, Test PyG graphs: {len(test_pyg)}')

    return train_nx, val_nx, test_nx, train_pyg, val_pyg, test_pyg