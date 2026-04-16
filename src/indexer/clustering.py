"""
Document Clustering using K-Means on document vectors.

CO3 Alignment: Classification and Clustering concepts.

Clustering groups similar documents together based on their
semantic vectors. This helps:
- Understand document collection structure
- Browse documents by topic
- Improve search by returning diverse results
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from src.indexer.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class DocumentClustering:
    """
    K-Means clustering on document-level semantic vectors.

    Documents are represented as mean vectors of their chunk embeddings,
    then grouped into K clusters using K-Means algorithm.
    """

    def __init__(self, vector_index: VectorIndex):
        self.vector_index = vector_index
        self.labels: Dict[int, int] = {}  # doc_id → cluster_id
        self.centroids: Optional[np.ndarray] = None
        self.k: int = 0
        self.cluster_names: Dict[int, str] = {}

    def cluster(self, k: int = 5) -> Dict[int, List[int]]:
        """
        Perform K-Means clustering on document vectors.

        Args:
            k: Number of clusters.

        Returns:
            Dict mapping cluster_id → list of doc_ids.

        CO3 Alignment: Applying clustering to document collection.
        """
        from sklearn.cluster import KMeans

        doc_ids, vectors = self.vector_index.get_all_doc_vectors()

        if len(doc_ids) == 0:
            logger.warning("No document vectors available for clustering")
            return {}

        # Adjust k if we have fewer documents
        k = min(k, len(doc_ids))
        self.k = k

        logger.info(f"Clustering {len(doc_ids)} documents into {k} clusters")

        # Run K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        self.centroids = kmeans.cluster_centers_

        # Build cluster assignments
        clusters: Dict[int, List[int]] = {i: [] for i in range(k)}
        for doc_id, label in zip(doc_ids, cluster_labels):
            self.labels[doc_id] = int(label)
            clusters[int(label)].append(doc_id)

        # Log cluster sizes
        for cid, docs in clusters.items():
            logger.info(f"  Cluster {cid}: {len(docs)} documents")

        return clusters

    def get_cluster(self, cluster_id: int) -> List[int]:
        """Get all doc_ids in a specific cluster."""
        return [
            doc_id
            for doc_id, label in self.labels.items()
            if label == cluster_id
        ]

    def get_document_cluster(self, doc_id: int) -> int:
        """Get the cluster assignment for a specific document."""
        return self.labels.get(doc_id, -1)

    def get_similar_documents(
        self, doc_id: int, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find documents most similar to a given document
        using vector distance within the same cluster.
        """
        target_vector = self.vector_index.get_doc_vector(doc_id)
        if target_vector is None:
            return []

        cluster_id = self.labels.get(doc_id, -1)
        candidates = self.get_cluster(cluster_id) if cluster_id >= 0 else list(
            self.labels.keys()
        )

        similarities = []
        for cand_id in candidates:
            if cand_id == doc_id:
                continue
            cand_vector = self.vector_index.get_doc_vector(cand_id)
            if cand_vector is not None:
                sim = float(np.dot(target_vector, cand_vector))
                similarities.append((cand_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_cluster_stats(self) -> List[dict]:
        """Get statistics for each cluster."""
        if not self.labels:
            return []

        clusters: Dict[int, List[int]] = {i: [] for i in range(self.k)}
        for doc_id, label in self.labels.items():
            clusters[label].append(doc_id)

        stats = []
        for cid in range(self.k):
            docs = clusters.get(cid, [])
            stats.append({
                "cluster_id": cid,
                "num_documents": len(docs),
                "doc_ids": docs,
            })

        return stats
