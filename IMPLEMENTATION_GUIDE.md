# Implementation Guide: Entity Resolution Performance Improvements
## Technical Specification for Approaches 1-3

---

## Quick Reference

| Approach | When to Implement | Expected F1 | Complexity | Prerequisites |
|----------|-------------------|-------------|------------|---------------|
| **HAC with Constraints** | Week 1 | 87-89% | Low-Medium | scipy, numpy |
| **Graph Transitive Closure** | Week 4 | 88-90% | Medium | networkx, HAC implemented |
| **Siamese Network** | Week 7 | 91-93% | High | PyTorch/TensorFlow, GPU access |

---

## Approach 1: Hierarchical Agglomerative Clustering (HAC)

### Installation

```bash
# Add to requirements.txt
scipy>=1.11.0
numpy>=1.24.0

# Install
pip install scipy numpy
```

### File Structure

```
backend/
├── app/
│   ├── services/
│   │   ├── name_matcher.py (existing)
│   │   ├── clustering/
│   │   │   ├── __init__.py
│   │   │   ├── hac_matcher.py (NEW)
│   │   │   ├── constraint_learner.py (NEW)
│   │   │   └── utils.py (NEW)
│   └── config/
│       └── settings.py (UPDATE)
├── tests/
│   └── test_hac_matcher.py (NEW)
└── requirements.txt (UPDATE)
```

### Implementation Steps

#### Step 1: Create HAC Matcher Module

Create file: `backend/app/services/clustering/hac_matcher.py`

```python
"""
Hierarchical Agglomerative Clustering with constraints.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConstrainedHACMatcher:
    """
    Hierarchical clustering with must-link and cannot-link constraints.

    Builds on existing NameMatcher for similarity scoring.
    """

    def __init__(self, base_matcher, linkage_method='average'):
        """
        Initialize HAC matcher.

        Args:
            base_matcher: Instance of NameMatcher for similarity scoring
            linkage_method: 'average', 'complete', 'single', or 'ward'
        """
        self.base_matcher = base_matcher
        self.linkage_method = linkage_method
        self.must_link_constraints = []
        self.cannot_link_constraints = []

        logger.info(f"Initialized HAC matcher with linkage method: {linkage_method}")

    def add_constraints(self, must_link: Optional[List[Tuple[str, str]]] = None,
                       cannot_link: Optional[List[Tuple[str, str]]] = None):
        """
        Add clustering constraints.

        Args:
            must_link: List of (name1, name2) tuples that must be in same cluster
            cannot_link: List of (name1, name2) tuples that cannot be in same cluster
        """
        if must_link:
            self.must_link_constraints.extend(must_link)
            logger.info(f"Added {len(must_link)} must-link constraints")

        if cannot_link:
            self.cannot_link_constraints.extend(cannot_link)
            logger.info(f"Added {len(cannot_link)} cannot-link constraints")

    def _build_distance_matrix(self, names: List[str]) -> np.ndarray:
        """
        Build pairwise distance matrix using base matcher.

        Distance = 1 - similarity (convert similarity to distance metric)

        Args:
            names: List of company names

        Returns:
            n x n distance matrix
        """
        n = len(names)
        distances = np.zeros((n, n))

        logger.info(f"Building distance matrix for {n} names ({n*(n-1)//2} pairs)")

        # Calculate pairwise distances
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names[i+1:], start=i+1):
                similarity = self.base_matcher.calculate_confidence(name1, name2)
                distance = 1.0 - similarity
                distances[i, j] = distance
                distances[j, i] = distance

        logger.info("Distance matrix built successfully")
        return distances

    def _enforce_constraints(self, distances: np.ndarray, names: List[str]) -> np.ndarray:
        """
        Modify distance matrix to enforce constraints.

        Must-link: Set distance to 0.0 (force clustering together)
        Cannot-link: Set distance to 1.0 (prevent clustering together)

        Args:
            distances: Original distance matrix
            names: List of company names

        Returns:
            Modified distance matrix
        """
        name_to_idx = {name: idx for idx, name in enumerate(names)}

        # Enforce must-link constraints
        for name1, name2 in self.must_link_constraints:
            if name1 in name_to_idx and name2 in name_to_idx:
                i, j = name_to_idx[name1], name_to_idx[name2]
                distances[i, j] = 0.0
                distances[j, i] = 0.0

        # Enforce cannot-link constraints
        for name1, name2 in self.cannot_link_constraints:
            if name1 in name_to_idx and name2 in name_to_idx:
                i, j = name_to_idx[name1], name_to_idx[name2]
                distances[i, j] = 1.0
                distances[j, i] = 1.0

        logger.info(f"Enforced {len(self.must_link_constraints)} must-link and "
                   f"{len(self.cannot_link_constraints)} cannot-link constraints")

        return distances

    def cluster(self, names: List[str], threshold: float = 0.15) -> List[List[str]]:
        """
        Perform hierarchical clustering.

        Args:
            names: List of company names to cluster
            threshold: Distance threshold for forming clusters (lower = more conservative)

        Returns:
            List of clusters, where each cluster is a list of names
        """
        if not names:
            return []

        if len(names) == 1:
            return [[names[0]]]

        # Build distance matrix
        distances = self._build_distance_matrix(names)

        # Enforce constraints
        distances = self._enforce_constraints(distances, names)

        # Convert to condensed form (required by scipy)
        condensed_distances = squareform(distances, checks=False)

        # Perform hierarchical clustering
        logger.info(f"Performing hierarchical clustering with method: {self.linkage_method}")
        Z = linkage(condensed_distances, method=self.linkage_method)

        # Cut dendrogram at threshold
        cluster_labels = fcluster(Z, threshold, criterion='distance')

        # Group names by cluster label
        clusters = {}
        for name, label in zip(names, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(name)

        result = list(clusters.values())
        logger.info(f"Created {len(result)} clusters from {len(names)} names "
                   f"({(1 - len(result)/len(names))*100:.1f}% reduction)")

        return result

    def auto_tune_threshold(self, names: List[str],
                           target_num_groups: Optional[int] = None,
                           method: str = 'binary_search') -> float:
        """
        Automatically find optimal threshold.

        Args:
            names: List of company names
            target_num_groups: Target number of clusters (if known)
            method: 'binary_search' or 'elbow'

        Returns:
            Optimal threshold value
        """
        # Build distance matrix
        distances = self._build_distance_matrix(names)
        distances = self._enforce_constraints(distances, names)
        condensed_distances = squareform(distances, checks=False)

        # Perform clustering
        Z = linkage(condensed_distances, method=self.linkage_method)

        if method == 'binary_search' and target_num_groups:
            # Binary search for threshold that produces target number of groups
            thresholds = np.linspace(0.05, 0.50, 100)
            best_threshold = None
            best_diff = float('inf')

            for t in thresholds:
                labels = fcluster(Z, t, criterion='distance')
                num_groups = len(set(labels))
                diff = abs(num_groups - target_num_groups)

                if diff < best_diff:
                    best_diff = diff
                    best_threshold = t

                if diff == 0:
                    break  # Exact match found

            logger.info(f"Auto-tuned threshold: {best_threshold:.4f} "
                       f"(produces {len(set(fcluster(Z, best_threshold, criterion='distance')))} groups)")
            return best_threshold

        elif method == 'elbow':
            # Elbow method: Find largest gap in linkage heights
            heights = Z[:, 2]
            gaps = np.diff(heights)
            elbow_idx = np.argmax(gaps)
            threshold = heights[elbow_idx]

            logger.info(f"Elbow method threshold: {threshold:.4f}")
            return threshold

        else:
            # Default threshold
            return 0.15
```

#### Step 2: Create Constraint Learner

Create file: `backend/app/services/clustering/constraint_learner.py`

```python
"""
Automatic constraint generation from external knowledge.
"""
from typing import List, Tuple, Set
from rapidfuzz import fuzz
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConstraintLearner:
    """
    Generates must-link and cannot-link constraints automatically.
    """

    def __init__(self, embedding_service=None):
        """
        Initialize constraint learner.

        Args:
            embedding_service: Optional embedding service for semantic similarity
        """
        self.embedding_service = embedding_service
        self.ticker_db = self._load_ticker_database()

    def _load_ticker_database(self) -> dict:
        """
        Load ticker symbol database.

        TODO: Integrate with external API (Alpha Vantage, Yahoo Finance, etc.)
        For now, returns hardcoded common tickers.
        """
        return {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta Platforms',
            'TSLA': 'Tesla Inc.',
            'NFLX': 'Netflix Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'WFC': 'Wells Fargo',
            'IBM': 'International Business Machines',
            'ORCL': 'Oracle Corporation',
            'CSCO': 'Cisco Systems',
            'INTC': 'Intel Corporation',
            # Add more as needed
        }

    def generate_must_link(self, names: List[str]) -> List[Tuple[str, str]]:
        """
        Generate must-link constraints from ticker symbols.

        Args:
            names: List of company names

        Returns:
            List of (name1, name2) tuples that must be clustered together
        """
        constraints = []
        name_set = set(names)

        # Match ticker symbols to full names
        for ticker, full_name in self.ticker_db.items():
            if ticker in name_set:
                # Find best match for full name
                for name in names:
                    if name == ticker:
                        continue
                    # High string similarity → likely the same company
                    if fuzz.partial_ratio(name.lower(), full_name.lower()) > 80:
                        constraints.append((ticker, name))
                        logger.debug(f"Must-link: {ticker} ↔ {name}")

        logger.info(f"Generated {len(constraints)} must-link constraints from ticker symbols")
        return constraints

    def generate_cannot_link(self, names: List[str], threshold: float = 0.60) -> List[Tuple[str, str]]:
        """
        Generate cannot-link constraints for ambiguous shared-word cases.

        Logic: High token overlap BUT low semantic similarity → likely different entities

        Args:
            names: List of company names
            threshold: Minimum semantic similarity to avoid cannot-link

        Returns:
            List of (name1, name2) tuples that cannot be in same cluster
        """
        constraints = []

        # Predefined ambiguous patterns
        ambiguous_patterns = [
            ('american express', 'american airlines'),
            ('american express', 'american standard'),
            ('delta air lines', 'delta dental'),
            ('delta air lines', 'delta faucet'),
            ('continental airlines', 'continental tire'),
            ('united airlines', 'united healthcare'),
            ('target corporation', 'target marketing'),
            ('apple inc', 'apple records'),
            ('amazon', 'amazon logistics'),  # Only if logistics is separate entity
        ]

        name_lower = [n.lower() for n in names]

        for pattern1, pattern2 in ambiguous_patterns:
            # Find names matching patterns
            matches1 = [names[i] for i, n in enumerate(name_lower) if pattern1 in n]
            matches2 = [names[i] for i, n in enumerate(name_lower) if pattern2 in n]

            for name1 in matches1:
                for name2 in matches2:
                    constraints.append((name1, name2))
                    logger.debug(f"Cannot-link: {name1} ↮ {name2}")

        # Use embeddings for additional cannot-link detection
        if self.embedding_service:
            for i, name1 in enumerate(names):
                for name2 in names[i+1:]:
                    token_sim = fuzz.token_set_ratio(name1, name2) / 100.0
                    semantic_sim = self.embedding_service.similarity(name1, name2)

                    # High token overlap but low semantic similarity
                    if token_sim > 0.70 and semantic_sim < threshold:
                        constraints.append((name1, name2))
                        logger.debug(f"Cannot-link (embedding): {name1} ↮ {name2} "
                                   f"(token={token_sim:.2f}, semantic={semantic_sim:.2f})")

        logger.info(f"Generated {len(constraints)} cannot-link constraints")
        return constraints
```

#### Step 3: Update Settings

Add to `backend/app/config/settings.py`:

```python
class Settings:
    # ... existing settings ...

    # HAC Clustering Configuration
    HAC_LINKAGE_METHOD: str = os.getenv("HAC_LINKAGE_METHOD", "average")  # 'average', 'complete', 'single', 'ward'
    HAC_DISTANCE_THRESHOLD: float = float(os.getenv("HAC_DISTANCE_THRESHOLD", "0.15"))
    HAC_AUTO_TUNE: bool = os.getenv("HAC_AUTO_TUNE", "True").lower() == "true"
    HAC_USE_CONSTRAINTS: bool = os.getenv("HAC_USE_CONSTRAINTS", "True").lower() == "true"
```

#### Step 4: Integration with NameMatcher

Update `backend/app/services/name_matcher.py`:

```python
# Add import
from app.services.clustering.hac_matcher import ConstrainedHACMatcher
from app.services.clustering.constraint_learner import ConstraintLearner

class NameMatcher:
    def __init__(self, similarity_threshold=None, use_adaptive_threshold=False,
                 embedding_mode=None, use_hac=False):
        # ... existing init code ...

        self.use_hac = use_hac
        if use_hac:
            self.hac_matcher = ConstrainedHACMatcher(self, linkage_method=settings.HAC_LINKAGE_METHOD)
            self.constraint_learner = ConstraintLearner(self.embedding_service)
            logger.info("Initialized HAC clustering mode")

    def process_names(self, names, filename="unknown"):
        # ... existing code until Step 2: Group similar names ...

        # Step 2: Group similar names
        if self.use_hac:
            # Use HAC clustering instead of greedy
            logger.info("Using HAC clustering mode")

            # Generate constraints
            if settings.HAC_USE_CONSTRAINTS:
                must_link = self.constraint_learner.generate_must_link(names)
                cannot_link = self.constraint_learner.generate_cannot_link(names)
                self.hac_matcher.add_constraints(must_link=must_link, cannot_link=cannot_link)

            # Auto-tune threshold or use configured value
            if settings.HAC_AUTO_TUNE:
                threshold = self.hac_matcher.auto_tune_threshold(names, target_num_groups=None)
            else:
                threshold = settings.HAC_DISTANCE_THRESHOLD

            groups = self.hac_matcher.cluster(names, threshold=threshold)
        else:
            # Use existing greedy clustering
            groups = self.group_similar_names(names, adaptive_thresholds, pairwise_data)

        # ... rest of existing code ...
```

#### Step 5: Update API Route

Update `backend/app/api/routes.py`:

```python
@router.post("/process")
async def process_companies(
    file: UploadFile = File(...),
    use_adaptive_threshold: bool = Query(default=False),
    use_hac: bool = Query(default=False),  # NEW parameter
    embedding_mode: str = Query(default=None)
):
    # ... existing validation ...

    # Create matcher with HAC option
    matcher = NameMatcher(
        use_adaptive_threshold=use_adaptive_threshold,
        use_hac=use_hac,
        embedding_mode=embedding_mode
    )

    # ... rest of existing code ...
```

#### Step 6: Testing

Create file: `backend/tests/test_hac_matcher.py`

```python
"""
Tests for HAC matcher.
"""
import pytest
from app.services.name_matcher import NameMatcher
from app.services.clustering.hac_matcher import ConstrainedHACMatcher


def test_hac_basic_clustering():
    """Test basic HAC clustering without constraints."""
    names = [
        "Apple Inc.",
        "Apple Computer Inc.",
        "Apple",
        "Microsoft Corporation",
        "Microsoft Corp",
        "Microsoft"
    ]

    matcher = NameMatcher(use_hac=True)
    result = matcher.process_names(names)

    # Should create 2 groups (Apple and Microsoft)
    assert result['summary']['total_groups_created'] == 2

    # Check group membership
    mappings = result['mappings']
    apple_canonical = [m['canonical_name'] for m in mappings if 'Apple' in m['original_name']]
    ms_canonical = [m['canonical_name'] for m in mappings if 'Microsoft' in m['original_name']]

    # All Apple names should map to same canonical
    assert len(set(apple_canonical)) == 1

    # All Microsoft names should map to same canonical
    assert len(set(ms_canonical)) == 1


def test_hac_with_must_link():
    """Test HAC with must-link constraints."""
    names = ["Apple Inc.", "AAPL", "Microsoft Corp"]

    matcher = NameMatcher(use_hac=True)
    matcher.hac_matcher.add_constraints(must_link=[("Apple Inc.", "AAPL")])

    result = matcher.process_names(names)
    mappings = result['mappings']

    # Apple Inc. and AAPL must be in same group
    apple_group = [m for m in mappings if m['original_name'] == "Apple Inc."][0]['group_id']
    aapl_group = [m for m in mappings if m['original_name'] == "AAPL"][0]['group_id']

    assert apple_group == aapl_group


def test_hac_with_cannot_link():
    """Test HAC with cannot-link constraints."""
    names = ["American Airlines", "American Express"]

    matcher = NameMatcher(use_hac=True)
    matcher.hac_matcher.add_constraints(cannot_link=[("American Airlines", "American Express")])

    result = matcher.process_names(names)
    mappings = result['mappings']

    # Must be in different groups
    aa_group = [m for m in mappings if m['original_name'] == "American Airlines"][0]['group_id']
    amex_group = [m for m in mappings if m['original_name'] == "American Express"][0]['group_id']

    assert aa_group != amex_group


def test_hac_auto_tune_threshold():
    """Test automatic threshold tuning."""
    names = ["Apple Inc."] * 10 + ["Microsoft Corp"] * 10

    matcher = NameMatcher(use_hac=True)

    # Should find threshold that produces ~2 groups
    threshold = matcher.hac_matcher.auto_tune_threshold(names, target_num_groups=2)

    assert 0.0 < threshold < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run tests:
```bash
cd backend
pytest tests/test_hac_matcher.py -v
```

---

## Approach 2: Graph-Based Transitive Closure

### Installation

```bash
# Add to requirements.txt
networkx>=3.1
matplotlib>=3.7.0  # For visualization (optional)

# Install
pip install networkx matplotlib
```

### Implementation Steps

#### Step 1: Create Graph Matcher Module

Create file: `backend/app/services/clustering/graph_matcher.py`

```python
"""
Graph-based clustering with transitive closure.
"""
from typing import List, Dict, Set, Tuple
import networkx as nx
from networkx.algorithms.community import louvain_communities
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphBasedMatcher:
    """
    Graph-based clustering using transitive closure and community detection.
    """

    def __init__(self, base_matcher, decay_factor=0.95):
        """
        Initialize graph matcher.

        Args:
            base_matcher: Instance of NameMatcher for similarity scoring
            decay_factor: Multiplier for transitive scores (0 < decay < 1)
        """
        self.base_matcher = base_matcher
        self.decay_factor = decay_factor
        logger.info(f"Initialized GraphBasedMatcher with decay factor: {decay_factor}")

    def build_graph(self, names: List[str], threshold: float) -> nx.Graph:
        """
        Build weighted similarity graph.

        Args:
            names: List of company names
            threshold: Minimum similarity to create edge

        Returns:
            NetworkX graph with similarity weights
        """
        G = nx.Graph()
        G.add_nodes_from(names)

        logger.info(f"Building graph for {len(names)} names (threshold: {threshold})")

        # Add edges for all pairs above threshold
        edge_count = 0
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                score = self.base_matcher.calculate_confidence(name1, name2)

                if score >= threshold:
                    G.add_edge(name1, name2, weight=score)
                    edge_count += 1

        logger.info(f"Created graph with {edge_count} edges (avg degree: {2*edge_count/len(names):.1f})")

        return G

    def add_transitive_edges(self, G: nx.Graph, threshold: float, max_iterations: int = 3) -> int:
        """
        Add transitive closure edges with decay.

        Finds paths A → B → C and adds edge A → C if transitive score >= threshold.

        Args:
            G: Input graph
            threshold: Minimum transitive score to add edge
            max_iterations: Maximum transitive closure iterations

        Returns:
            Total number of edges added
        """
        total_added = 0

        for iteration in range(max_iterations):
            new_edges = []

            # Find 2-hop paths
            for node in G.nodes():
                neighbors_1hop = set(G.neighbors(node))

                for neighbor in neighbors_1hop:
                    neighbors_2hop = set(G.neighbors(neighbor)) - {node}

                    for target in neighbors_2hop:
                        if G.has_edge(node, target):
                            continue  # Edge already exists

                        # Calculate transitive score
                        score_1 = G[node][neighbor]['weight']
                        score_2 = G[neighbor][target]['weight']
                        transitive_score = min(score_1, score_2) * self.decay_factor

                        if transitive_score >= threshold:
                            new_edges.append((node, target, transitive_score))

            # Add new edges
            edges_added_this_iter = 0
            for node1, node2, score in new_edges:
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=score)
                    edges_added_this_iter += 1
                else:
                    # Strengthen existing edge
                    G[node1][node2]['weight'] = max(G[node1][node2]['weight'], score)

            total_added += edges_added_this_iter
            logger.info(f"Iteration {iteration + 1}: Added {edges_added_this_iter} transitive edges")

            if edges_added_this_iter == 0:
                break  # Converged

        logger.info(f"Total transitive edges added: {total_added}")
        return total_added

    def cluster_graph(self, G: nx.Graph, method: str = 'louvain') -> List[List[str]]:
        """
        Apply community detection algorithm.

        Args:
            G: Input graph
            method: 'louvain', 'connected_components', or 'label_propagation'

        Returns:
            List of clusters
        """
        logger.info(f"Clustering graph using method: {method}")

        if method == 'louvain':
            communities = louvain_communities(G, weight='weight', seed=42)
            clusters = [list(c) for c in communities]

        elif method == 'connected_components':
            components = nx.connected_components(G)
            clusters = [list(c) for c in components]

        elif method == 'label_propagation':
            from networkx.algorithms.community import label_propagation_communities
            communities = label_propagation_communities(G)
            clusters = [list(c) for c in communities]

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logger.info(f"Created {len(clusters)} clusters")
        return clusters

    def process_names(self, names: List[str], threshold: float = 0.85,
                     clustering_method: str = 'louvain') -> List[List[str]]:
        """
        Main pipeline: build graph → transitive closure → community detection.

        Args:
            names: List of company names
            threshold: Similarity threshold for edges
            clustering_method: Community detection algorithm

        Returns:
            List of clusters
        """
        # Build initial graph
        G = self.build_graph(names, threshold)

        # Add transitive edges
        self.add_transitive_edges(G, threshold)

        # Cluster using community detection
        clusters = self.cluster_graph(G, method=clustering_method)

        return clusters

    def visualize_graph(self, G: nx.Graph, output_path: str = "graph_viz.png"):
        """
        Visualize graph structure (useful for debugging).

        Args:
            G: Graph to visualize
            output_path: Output file path
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Layout
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')

            # Draw edges (thickness based on weight)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], alpha=0.6)

            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8)

            plt.title("Company Name Similarity Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Graph visualization saved to {output_path}")

        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")
```

#### Step 2: Integration

Update `backend/app/services/name_matcher.py`:

```python
# Add import
from app.services.clustering.graph_matcher import GraphBasedMatcher

class NameMatcher:
    def __init__(self, ..., use_graph=False):
        # ... existing init ...

        self.use_graph = use_graph
        if use_graph:
            self.graph_matcher = GraphBasedMatcher(self)
            logger.info("Initialized graph-based clustering mode")

    def process_names(self, names, filename="unknown"):
        # ... existing code ...

        # Step 2: Group similar names
        if self.use_graph:
            logger.info("Using graph-based clustering mode")
            threshold = self.similarity_threshold / 100.0
            groups = self.graph_matcher.process_names(names, threshold=threshold)
        elif self.use_hac:
            # ... HAC code ...
        else:
            # ... existing greedy code ...

        # ... rest of code ...
```

---

## Approach 3: Siamese Neural Network

### Installation

```bash
# Add to requirements.txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.2
datasets>=2.13.0

# Install (with CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers datasets
```

### Implementation Steps

*Due to length constraints, see detailed implementation in separate document: `SIAMESE_NETWORK_GUIDE.md`*

**Quick Start:**
1. Data preparation: Augment ground truth to 15K pairs
2. Model architecture: Sentence-BERT + fine-tuning layers
3. Training: 20 epochs with contrastive loss
4. Inference: Replace OpenAI embeddings with fine-tuned model

---

## Testing & Validation

### Performance Testing Script

Create file: `backend/test_clustering_approaches.py`

```python
"""
Compare clustering approaches on ground truth data.
"""
import pandas as pd
import time
from app.services.name_matcher import NameMatcher
from sklearn.metrics import precision_score, recall_score, f1_score


def load_ground_truth():
    """Load ground truth data."""
    df = pd.read_csv('ground_truth.csv')
    return df


def evaluate_clustering(predicted_groups, ground_truth_df):
    """
    Evaluate clustering quality using pairwise metrics.

    Args:
        predicted_groups: List of predicted clusters
        ground_truth_df: DataFrame with 'Original Name' and 'Canonical Name' columns

    Returns:
        Dict with precision, recall, F1
    """
    # Build ground truth pairs
    gt_pairs = set()
    gt_groups = ground_truth_df.groupby('Canonical Name')['Original Name'].apply(list).tolist()

    for group in gt_groups:
        for i, name1 in enumerate(group):
            for name2 in group[i+1:]:
                gt_pairs.add((min(name1, name2), max(name1, name2)))

    # Build predicted pairs
    pred_pairs = set()
    for group in predicted_groups:
        for i, name1 in enumerate(group):
            for name2 in group[i+1:]:
                pred_pairs.add((min(name1, name2), max(name1, name2)))

    # Calculate metrics
    true_positives = len(gt_pairs & pred_pairs)
    false_positives = len(pred_pairs - gt_pairs)
    false_negatives = len(gt_pairs - pred_pairs)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def test_approach(name, matcher_kwargs):
    """Test a clustering approach."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    # Load data
    df = load_ground_truth()
    names = df['Original Name'].tolist()

    # Create matcher
    matcher = NameMatcher(**matcher_kwargs)

    # Process
    start_time = time.time()
    result = matcher.process_names(names, filename='ground_truth.csv')
    elapsed = time.time() - start_time

    # Extract groups
    groups = []
    group_map = {}
    for mapping in result['mappings']:
        group_id = mapping['group_id']
        if group_id not in group_map:
            group_map[group_id] = []
        group_map[group_id].append(mapping['original_name'])
    groups = list(group_map.values())

    # Evaluate
    metrics = evaluate_clustering(groups, df)

    # Print results
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"\nCounts:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"\nClusters:")
    print(f"  Predicted Groups: {len(groups)}")
    print(f"  Ground Truth Groups: {df['Canonical Name'].nunique()}")
    print(f"\nTiming:")
    print(f"  Processing Time: {elapsed:.2f}s")

    return metrics, elapsed


if __name__ == "__main__":
    approaches = [
        ("Baseline (Current System)", {
            'use_adaptive_threshold': True,
            'embedding_mode': 'openai-large'
        }),
        ("HAC with Constraints", {
            'use_hac': True,
            'embedding_mode': 'openai-large'
        }),
        ("Graph-Based Transitive Closure", {
            'use_graph': True,
            'embedding_mode': 'openai-large'
        }),
        # Add Siamese when ready
    ]

    results = []
    for name, kwargs in approaches:
        metrics, elapsed = test_approach(name, kwargs)
        results.append((name, metrics, elapsed))

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Approach':<40} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time':>8}")
    print(f"{'-'*80}")

    for name, metrics, elapsed in results:
        print(f"{name:<40} {metrics['f1']:>8.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>8.4f} {elapsed:>7.2f}s")
```

Run comparison:
```bash
cd backend
python test_clustering_approaches.py
```

---

## Configuration & Deployment

### Environment Variables

Add to `backend/.env`:

```bash
# HAC Configuration
HAC_LINKAGE_METHOD=average
HAC_DISTANCE_THRESHOLD=0.15
HAC_AUTO_TUNE=true
HAC_USE_CONSTRAINTS=true

# Graph Configuration
GRAPH_DECAY_FACTOR=0.95
GRAPH_MAX_ITERATIONS=3
GRAPH_CLUSTERING_METHOD=louvain

# Siamese Network Configuration (when ready)
SIAMESE_MODEL_PATH=models/siamese_company_matcher.pth
SIAMESE_BATCH_SIZE=32
SIAMESE_USE_GPU=true
```

### API Usage Examples

```bash
# Use HAC clustering
curl -X POST "http://localhost:8000/api/process?use_hac=true" \
  -F "file=@companies.csv"

# Use graph-based clustering
curl -X POST "http://localhost:8000/api/process?use_graph=true" \
  -F "file=@companies.csv"

# Combine approaches (HAC + Graph refinement)
curl -X POST "http://localhost:8000/api/process?use_hac=true&use_graph=true" \
  -F "file=@companies.csv"
```

---

## Monitoring & Debugging

### Logging

All approaches include detailed logging:

```python
# Enable debug logging in .env
LOG_LEVEL=DEBUG

# View logs
tail -f backend/logs/app.log
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile HAC
matcher = NameMatcher(use_hac=True)
cProfile.run('matcher.process_names(names)', 'hac_profile.stats')

# Analyze
p = pstats.Stats('hac_profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def test_large_dataset():
    matcher = NameMatcher(use_hac=True)
    names = ["Company " + str(i) for i in range(10000)]
    matcher.process_names(names)

test_large_dataset()
```

---

## Troubleshooting

### Common Issues

**Issue 1: HAC too slow for large datasets (>5000 names)**
- **Solution:** Use FastCluster library (100x faster)
  ```bash
  pip install fastcluster
  ```
  ```python
  from fastcluster import linkage  # Drop-in replacement
  ```

**Issue 2: Graph approach runs out of memory**
- **Solution:** Use sparse matrices
  ```python
  from scipy.sparse import csr_matrix
  # Store only non-zero edges
  ```

**Issue 3: Constraints conflict (must-link + cannot-link for same pair)**
- **Solution:** Add validation in `add_constraints()`
  ```python
  def _validate_constraints(self):
      must_set = set(self.must_link_constraints)
      cannot_set = set(self.cannot_link_constraints)
      conflicts = must_set & cannot_set
      if conflicts:
          raise ValueError(f"Conflicting constraints: {conflicts}")
  ```

---

## Next Steps

1. **Week 1-3:** Implement & test HAC with constraints
2. **Week 4-6:** Add graph-based transitive closure
3. **Week 7-14:** Develop Siamese network (see separate guide)
4. **Week 15-16:** Ensemble integration & polish

Good luck with implementation!
