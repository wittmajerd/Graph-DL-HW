import numpy as np
import networkx as nx
import hashlib
from typing import List, Dict, Optional
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WeisfeilerLehmanHashing:
    """
    Weisfeiler-Lehman feature extraction for graph representation.
    """
    def __init__(
        self, 
        graph: nx.Graph, 
        wl_iterations: int = 2,
        use_node_attribute: Optional[str] = None,
        erase_base_features: bool = False
    ):
        """
        Initialize the Weisfeiler-Lehman feature extractor.
        
        Args:
            graph: A NetworkX graph object.
            wl_iterations: Number of Weisfeiler-Lehman iterations.
            use_node_attribute: Node attribute to use as initial feature.
            erase_base_features: Whether to only use WL features without base features.
        """
        self.graph = graph
        self.wl_iterations = wl_iterations
        self.use_node_attribute = use_node_attribute
        self.erase_base_features = erase_base_features
        
        # Extract features
        self._extracted_features = []
        self._extract_features()
    
    def _extract_features(self):
        # Get initial features
        features = self._get_initial_features()
        
        # Store base features unless erasing is requested
        if not self.erase_base_features:
            self._extracted_features = [str(v) for _, v in features.items()]
        
        # Do WL iterations
        for _ in range(self.wl_iterations):
            features = self._do_wl_recursion(features)
    
    def _get_initial_features(self) -> Dict:
        """Extract initial node features from the graph."""
        if self.use_node_attribute is not None:
            features = {}
            for node in self.graph.nodes():
                if self.use_node_attribute in self.graph.nodes[node]:
                    features[node] = self.graph.nodes[node][self.use_node_attribute]
                else:
                    features[node] = self.graph.degree(node)
        else:
            features = {node: self.graph.degree(node) for node in self.graph.nodes()}
        
        return features
    
    def _do_wl_recursion(self, features: Dict) -> Dict:
        """Perform one iteration of Weisfeiler-Lehman relabeling."""
        new_features = {}
        
        for node in self.graph.nodes():
            # Get current node label and neighbor labels
            neighbors = list(self.graph.neighbors(node))
            neighbor_labels = [features[neighbor] for neighbor in neighbors]
            
            # Create WL feature as a concatenation of current and sorted neighbor labels
            wl_feature = [str(features[node])] + sorted([str(label) for label in neighbor_labels])
            wl_feature_str = "_".join(wl_feature)
            
            # Hash feature string to get a new label
            hash_obj = hashlib.md5(wl_feature_str.encode())
            hashed_feature = hash_obj.hexdigest()
            
            new_features[node] = hashed_feature
        
        # Add newly computed features to the feature list
        self._extracted_features.extend(list(new_features.values()))
        
        return new_features
    
    def get_graph_features(self) -> List[str]:
        """Get the extracted WL features for the graph."""
        return self._extracted_features


class Graph2Vec:
    """
    Implementation of Graph2Vec algorithm for learning distributed representations of graphs.
    
    This implementation is designed to work with NetworkX graphs and without requiring 
    the karateclub package, making it compatible with PyG's QM9 dataset.
    """
    
    def __init__(
        self,
        wl_iterations: int = 2,
        use_node_attribute: Optional[str] = None,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):
        """
        Initialize the Graph2Vec model.
        
        Args:
            wl_iterations: Number of Weisfeiler-Lehman iterations.
            use_node_attribute: Node attribute to use as initial feature.
            dimensions: Dimensionality of graph embeddings.
            workers: Number of worker threads for training.
            down_sampling: Down sampling rate for frequent features.
            epochs: Number of training epochs.
            learning_rate: Initial learning rate.
            min_count: Minimum count of features.
            seed: Random seed for reproducibility.
            erase_base_features: Whether to erase base features.
        """
        self.wl_iterations = wl_iterations
        self.use_node_attribute = use_node_attribute
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features
        
        self.model = None
        self._embedding = None
    
    def _set_seed(self):
        """Set random seed for reproducibility."""
        np.random.seed(self.seed)
    
    def _check_graphs(self, graphs):
        """Ensure input graphs are valid NetworkX graphs."""
        for i, graph in enumerate(graphs):
            if not isinstance(graph, nx.Graph):
                raise TypeError(f"Graph at index {i} is not a NetworkX Graph object.")
        return graphs
    
    def fit(self, graphs: List[nx.Graph]):
        """
        Fit the Graph2Vec model on the provided graphs.
        
        Args:
            graphs: List of NetworkX graph objects.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        
        # Extract features using Weisfeiler-Lehman algorithm
        documents = []
        for i, graph in enumerate(graphs):
            wl_hash = WeisfeilerLehmanHashing(
                graph=graph,
                wl_iterations=self.wl_iterations,
                use_node_attribute=self.use_node_attribute,
                erase_base_features=self.erase_base_features
            )
            documents.append(TaggedDocument(words=wl_hash.get_graph_features(), tags=[str(i)]))
        
        # Train Doc2Vec model
        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )
        
        # Store embeddings
        self._embedding = np.array([self.model.dv[str(i)] for i in range(len(documents))])
    
    def get_embedding(self) -> np.ndarray:
        """
        Get the computed graph embeddings.
        
        Returns:
            A numpy array of shape (n_graphs, dimensions) containing graph embeddings.
        """
        if self._embedding is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self._embedding
    
    def infer(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Infer embeddings for new graphs using the trained model.
        
        Args:
            graphs: List of NetworkX graph objects.
            
        Returns:
            A numpy array of shape (n_graphs, dimensions) containing inferred embeddings.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        self._set_seed()
        graphs = self._check_graphs(graphs)
        
        # Extract features using WL algorithm
        documents = []
        for graph in graphs:
            wl_hash = WeisfeilerLehmanHashing(
                graph=graph,
                wl_iterations=self.wl_iterations,
                use_node_attribute=self.use_node_attribute,
                erase_base_features=self.erase_base_features
            )
            documents.append(wl_hash.get_graph_features())
        
        # Infer vectors for new documents
        embeddings = np.array([
            self.model.infer_vector(
                doc,
                alpha=self.learning_rate,
                min_alpha=0.00001,
                epochs=self.epochs
            ) for doc in documents
        ])
        
        return embeddings
    
    def save(self, path: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str):
        """Load a pre-trained model from disk."""
        instance = cls()
        instance.model = Doc2Vec.load(path)
        return instance