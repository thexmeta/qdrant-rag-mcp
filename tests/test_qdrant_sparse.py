from qdrant_client import QdrantClient
from qdrant_client.models import SparseVectorParams, Modifier

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="test_sparse",
    vectors_config={},
    sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}
)
print("Created successfully with IDF")
