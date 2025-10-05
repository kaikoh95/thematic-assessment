"""
Text Clustering Module

UMAP + HDBSCAN pipeline for semantic clustering of sentence embeddings.
Includes KMeans fallback for edge cases.

Key features:
- Dimensionality reduction (384→5-10 dims) with UMAP
- Density-based clustering with HDBSCAN
- Automatic parameter adaptation
- KMeans fallback for high-noise scenarios
- Quality metrics (silhouette, noise ratio)

Based on latest research (2024-2025):
- Use scikit-learn's HDBSCAN (Python 3.12 compatible)
- UMAP essential for high-dim embeddings (>50 dims)
- Cosine metric for UMAP, Euclidean for HDBSCAN
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

logger = logging.getLogger(__name__)


class TextClusterer:
    """
    Production-ready text clustering with UMAP + HDBSCAN

    Pipeline:
    1. UMAP: Reduce embeddings from 384 → 5-10 dimensions
    2. HDBSCAN: Density-based clustering
    3. Quality Check: Noise ratio, cluster count
    4. Fallback: KMeans if HDBSCAN produces poor results

    Usage:
        clusterer = TextClusterer()
        result = clusterer.cluster(embeddings, sentences)
    """

    def __init__(
        self,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 30,
        min_cluster_size: Optional[int] = None,
        max_clusters: int = 10,
        noise_threshold: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize clusterer with parameters

        Args:
            umap_n_components: Target dimensions (5-10 recommended)
            umap_n_neighbors: UMAP neighbors (15-30 for clustering)
            min_cluster_size: Min samples per cluster (None = adaptive)
            max_clusters: Maximum clusters to return
            noise_threshold: Max acceptable noise ratio (0-1)
            random_state: Random seed for reproducibility
        """
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.noise_threshold = noise_threshold
        self.random_state = random_state

    def cluster(
        self,
        embeddings: np.ndarray,
        sentences: Optional[List[str]] = None
    ) -> Dict:
        """
        Cluster sentence embeddings using UMAP + HDBSCAN

        Args:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
            sentences: Optional list of original sentences (for logging)

        Returns:
            {
                'labels': np.ndarray,          # Cluster assignments
                'reduced_embeddings': np.ndarray,  # UMAP-reduced embeddings
                'n_clusters': int,             # Number of clusters found
                'noise_count': int,            # Number of noise points
                'noise_ratio': float,          # Proportion of noise
                'algorithm': str,              # 'hdbscan' or 'kmeans'
                'quality_score': float,        # Silhouette score
                'metadata': dict              # Additional info
            }
        """
        n_samples = len(embeddings)

        if n_samples < 3:
            logger.warning(f"Too few samples ({n_samples}) for clustering")
            return self._single_cluster_result(embeddings, n_samples)

        logger.info(
            f"Clustering {n_samples} samples "
            f"(umap_dims={self.umap_n_components}, max_clusters={self.max_clusters})"
        )

        # Step 1: UMAP dimensionality reduction
        reduced_embeddings = self._umap_reduce(embeddings)

        # Step 2: Try HDBSCAN first
        hdbscan_result = self._try_hdbscan(reduced_embeddings, n_samples)

        # Step 3: Check if HDBSCAN results are acceptable
        if self._is_acceptable_clustering(hdbscan_result):
            logger.info(
                f"HDBSCAN succeeded: {hdbscan_result['n_clusters']} clusters, "
                f"{hdbscan_result['noise_ratio']:.1%} noise"
            )
            hdbscan_result['reduced_embeddings'] = reduced_embeddings
            return hdbscan_result

        # Step 4: Fallback to KMeans
        logger.warning(
            f"HDBSCAN produced poor results (noise={hdbscan_result['noise_ratio']:.1%}), "
            f"falling back to KMeans"
        )
        kmeans_result = self._fallback_kmeans(reduced_embeddings, n_samples)
        kmeans_result['reduced_embeddings'] = reduced_embeddings

        return kmeans_result

    def _umap_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embedding dimensionality using UMAP

        Critical for HDBSCAN: performs poorly on high-dim data (>50 dims)

        Args:
            embeddings: High-dimensional embeddings (e.g., 384 dims)

        Returns:
            Reduced embeddings (5-10 dims)
        """
        n_samples = len(embeddings)
        n_features = embeddings.shape[1]

        # Adaptive n_components (can't exceed min(n_samples, n_features))
        n_components = min(self.umap_n_components, n_samples, n_features)

        # Adaptive n_neighbors (can't exceed n_samples - 1)
        n_neighbors = min(self.umap_n_neighbors, n_samples - 1)

        logger.info(
            f"UMAP: reducing {embeddings.shape[1]}→{n_components} dims "
            f"(n_neighbors={n_neighbors})"
        )

        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.0,  # Tight packing for clustering
                metric='cosine',  # Best for sentence embeddings
                random_state=self.random_state
            )

            reduced = reducer.fit_transform(embeddings)
            logger.info(f"UMAP completed: shape={reduced.shape}")

            return reduced

        except Exception as e:
            logger.error(f"UMAP failed: {e}, using PCA fallback")
            # Fallback to PCA if UMAP fails
            from sklearn.decomposition import PCA
            # Adaptive n_components for PCA as well
            pca_components = min(self.umap_n_components, n_samples, n_features)
            pca = PCA(n_components=pca_components, random_state=self.random_state)
            return pca.fit_transform(embeddings)

    def _try_hdbscan(self, embeddings: np.ndarray, n_samples: int) -> Dict:
        """
        Attempt clustering with HDBSCAN

        Args:
            embeddings: UMAP-reduced embeddings
            n_samples: Number of samples

        Returns:
            Clustering results dictionary
        """
        # Adaptive min_cluster_size
        if self.min_cluster_size is not None:
            min_cluster_size = self.min_cluster_size
        else:
            min_cluster_size = self._adaptive_min_cluster_size(n_samples)

        # min_samples: more conservative than min_cluster_size
        min_samples = max(2, min_cluster_size // 2)

        logger.info(
            f"HDBSCAN: min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}"
        )

        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',  # After UMAP, euclidean is appropriate
                cluster_selection_method='eom',  # Excess of Mass (stable)
                algorithm='auto'  # Auto-select fastest algorithm
            )

            labels = clusterer.fit_predict(embeddings)

            # Calculate metrics
            n_clusters, noise_ratio = self._calculate_cluster_stats(labels)

            # Quality score
            quality_score = self._calculate_quality_score(embeddings, labels)

            return {
                'labels': labels,
                'n_clusters': n_clusters,
                'noise_count': int(list(labels).count(-1)),
                'noise_ratio': noise_ratio,
                'algorithm': 'hdbscan',
                'quality_score': quality_score,
                'metadata': {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples
                }
            }

        except Exception as e:
            logger.error(f"HDBSCAN failed: {e}")
            # Return poor result to trigger fallback
            return {
                'labels': np.full(n_samples, -1),  # All noise
                'n_clusters': 0,
                'noise_count': n_samples,
                'noise_ratio': 1.0,
                'algorithm': 'hdbscan_failed',
                'quality_score': -1.0,
                'metadata': {'error': str(e)}
            }

    def _fallback_kmeans(self, embeddings: np.ndarray, n_samples: int) -> Dict:
        """
        Fallback to KMeans clustering

        Used when HDBSCAN produces excessive noise or poor results.

        Args:
            embeddings: UMAP-reduced embeddings
            n_samples: Number of samples

        Returns:
            Clustering results dictionary
        """
        # Estimate optimal K
        optimal_k = self._estimate_optimal_k(
            embeddings,
            k_range=range(3, min(self.max_clusters + 1, n_samples // 2))
        )

        logger.info(f"KMeans: using k={optimal_k} clusters")

        try:
            kmeans = KMeans(
                n_clusters=optimal_k,
                n_init=10,
                random_state=self.random_state,
                max_iter=300
            )

            labels = kmeans.fit_predict(embeddings)

            # Calculate metrics
            n_clusters, _ = self._calculate_cluster_stats(labels)
            quality_score = silhouette_score(embeddings, labels)

            return {
                'labels': labels,
                'n_clusters': n_clusters,
                'noise_count': 0,  # KMeans assigns all points
                'noise_ratio': 0.0,
                'algorithm': 'kmeans',
                'quality_score': quality_score,
                'metadata': {
                    'k': optimal_k,
                    'reason': 'hdbscan_fallback'
                }
            }

        except Exception as e:
            logger.error(f"KMeans failed: {e}")
            # Last resort: single cluster
            return self._single_cluster_result(embeddings, n_samples)

    def _adaptive_min_cluster_size(self, n_samples: int) -> int:
        """
        Determine min_cluster_size based on dataset size

        Guidelines from research:
        - < 50 samples: 3
        - 50-200: 5-10 (n/30)
        - 200-500: 10-15 (n/40)
        - > 500: 15-25 (n/40)
        """
        if n_samples < 50:
            return 3
        elif n_samples < 200:
            return max(5, n_samples // 30)
        else:
            return max(10, n_samples // 40)

    def _estimate_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: range
    ) -> int:
        """
        Estimate optimal K for KMeans using silhouette score

        Args:
            embeddings: Reduced embeddings
            k_range: Range of K values to try

        Returns:
            Optimal K value
        """
        best_k = 3
        best_score = -1

        for k in k_range:
            if k >= len(embeddings):
                break

            try:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logger.warning(f"K={k} failed: {e}")
                continue

        logger.info(f"Optimal K={best_k} (silhouette={best_score:.3f})")
        return best_k

    def _calculate_cluster_stats(self, labels: np.ndarray) -> Tuple[int, float]:
        """
        Calculate cluster statistics

        Args:
            labels: Cluster assignments

        Returns:
            (n_clusters, noise_ratio)
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = list(labels).count(-1)
        noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0.0

        return n_clusters, noise_ratio

    def _calculate_quality_score(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Calculate clustering quality score (silhouette)

        Args:
            embeddings: Reduced embeddings
            labels: Cluster assignments

        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        # Filter out noise points
        valid_mask = labels != -1

        if np.sum(valid_mask) < 2:
            return -1.0

        n_clusters = len(set(labels[valid_mask]))

        if n_clusters < 2:
            return -1.0

        try:
            return silhouette_score(embeddings[valid_mask], labels[valid_mask])
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return -1.0

    def _is_acceptable_clustering(self, result: Dict) -> bool:
        """
        Check if clustering result is acceptable

        Criteria:
        - Noise ratio < threshold
        - Cluster count in reasonable range (3-max_clusters)
        - Quality score > 0 (better than random)

        Args:
            result: Clustering result dictionary

        Returns:
            True if acceptable, False otherwise
        """
        return (
            result['noise_ratio'] < self.noise_threshold and
            3 <= result['n_clusters'] <= self.max_clusters and
            result['quality_score'] > 0.0
        )

    def _single_cluster_result(
        self,
        embeddings: np.ndarray,
        n_samples: int
    ) -> Dict:
        """
        Fallback: assign all samples to a single cluster

        Used when everything else fails or dataset is too small.

        Args:
            embeddings: Embeddings
            n_samples: Number of samples

        Returns:
            Single cluster result
        """
        logger.warning("Using single cluster fallback")

        return {
            'labels': np.zeros(n_samples, dtype=int),
            'reduced_embeddings': embeddings,
            'n_clusters': 1,
            'noise_count': 0,
            'noise_ratio': 0.0,
            'algorithm': 'single_cluster_fallback',
            'quality_score': 0.0,
            'metadata': {'reason': 'too_few_samples_or_failure'}
        }

    def assign_noise_to_nearest_cluster(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Assign noise points to their nearest cluster

        Args:
            labels: Cluster assignments (may contain -1 for noise)
            embeddings: Reduced embeddings

        Returns:
            Updated labels with no noise points
        """
        noise_mask = labels == -1

        if not np.any(noise_mask):
            return labels  # No noise to reassign

        logger.info(f"Reassigning {np.sum(noise_mask)} noise points to nearest cluster")

        # Get noise points and clustered points
        noise_points = embeddings[noise_mask]
        clustered_points = embeddings[~noise_mask]
        clustered_labels = labels[~noise_mask]

        # Find nearest clustered point for each noise point
        distances = pairwise_distances(noise_points, clustered_points, metric='euclidean')
        nearest_indices = np.argmin(distances, axis=1)

        # Assign noise to nearest cluster
        updated_labels = labels.copy()
        updated_labels[noise_mask] = clustered_labels[nearest_indices]

        return updated_labels


if __name__ == "__main__":
    # Example usage and testing
    import time
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic embeddings
    np.random.seed(42)
    n_samples = 100
    embedding_dim = 384

    # Create 3 clusters with some noise
    cluster1 = np.random.randn(30, embedding_dim) + [1, 0] + [0] * (embedding_dim - 2)
    cluster2 = np.random.randn(30, embedding_dim) + [0, 1] + [0] * (embedding_dim - 2)
    cluster3 = np.random.randn(30, embedding_dim) + [-1, -1] + [0] * (embedding_dim - 2)
    noise = np.random.randn(10, embedding_dim)

    embeddings = np.vstack([cluster1, cluster2, cluster3, noise])

    print("=" * 60)
    print("Text Clustering Test")
    print("=" * 60)
    print(f"Samples: {n_samples}")
    print(f"Embedding dimension: {embedding_dim}")

    # Test clustering
    clusterer = TextClusterer(
        umap_n_components=5,
        max_clusters=10,
        noise_threshold=0.5
    )

    start = time.time()
    result = clusterer.cluster(embeddings)
    duration = time.time() - start

    print(f"\nClustering completed in {duration:.3f}s")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Clusters found: {result['n_clusters']}")
    print(f"Noise points: {result['noise_count']} ({result['noise_ratio']:.1%})")
    print(f"Quality score: {result['quality_score']:.3f}")
    print(f"Reduced embedding shape: {result['reduced_embeddings'].shape}")

    # Test noise reassignment
    if result['noise_count'] > 0:
        print("\nReassigning noise points...")
        updated_labels = clusterer.assign_noise_to_nearest_cluster(
            result['labels'],
            result['reduced_embeddings']
        )
        print(f"Noise points after reassignment: {list(updated_labels).count(-1)}")
