
import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):
        """
        FAISS vector store using inner-product similarity.

        Embeddings are normalized before being added,
        so inner product is equivalent to cosine similarity.
        """

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        # Store the original documents/chunks
        # corresponding to FAISS vectors.
        self.documents = []

    def add(
        self,
        embeddings,
        documents
    ):
        """
        Add embeddings and their corresponding
        documents to the FAISS index.
        """

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(
                1,
                -1
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if len(vectors) != len(documents):
            raise ValueError(
                "Number of embeddings must match "
                "number of documents"
            )

        self.index.add(vectors)

        self.documents.extend(
            documents
        )

    def search(
        self,
        query_embedding,
        top_k: int = 5
    ):
        """
        Search for the most semantically similar
        documents.
        """

        if not self.documents:
            return []

        query_vector = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(
                1,
                -1
            )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension "
                f"{self.dimension}, "
                f"got {query_vector.shape[1]}"
            )

        k = min(
            top_k,
            len(self.documents)
        )

        scores, indices = self.index.search(
            query_vector,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "score": float(score),
                "document": self.documents[index]
            })

        return results