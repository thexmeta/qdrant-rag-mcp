with open('src/qdrant_mcp_context_aware.py', 'r') as f:
    content = f.read()

content = content.replace("from utils.embeddings import get_embeddings_manager", "from utils.embeddings import get_embeddings_manager\nfrom utils.sparse_embeddings import get_sparse_embeddings_manager")

with open('src/qdrant_mcp_context_aware.py', 'w') as f:
    f.write(content)
