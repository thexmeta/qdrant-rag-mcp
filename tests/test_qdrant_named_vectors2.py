from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="test_unnamed",
    vectors_config={"": VectorParams(size=4, distance=Distance.COSINE)}
)

client.upsert(
    collection_name="test_unnamed",
    points=[PointStruct(id=1, vector=[1.0, 2.0, 3.0, 4.0], payload={})]
)
print("Upserted successfully with unnamed vector array")

client.create_collection(
    collection_name="test_unnamed2",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE)
)

client.upsert(
    collection_name="test_unnamed2",
    points=[PointStruct(id=1, vector=[1.0, 2.0, 3.0, 4.0], payload={})]
)
print("Upserted successfully with plain VectorParams array")
