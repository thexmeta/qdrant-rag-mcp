from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams

client = QdrantClient(":memory:")

try:
    client.create_collection(
        collection_name="test_named",
        vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
    )
    print("Created named 'dense' successfully")
except Exception as e:
    print("Error:", e)

try:
    client.create_collection(
        collection_name="test_unnamed",
        vectors_config={"": VectorParams(size=4, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
    )
    print("Created unnamed '' successfully")
except Exception as e:
    print("Error:", e)
