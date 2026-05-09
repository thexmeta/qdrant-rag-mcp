import re

with open('src/config.py', 'r') as f:
    content = f.read()

pattern = r'env_mappings = \{'
replacement = '''env_mappings = {
            "hybrid_search.sparse_method": "QDRANT_SPARSE_METHOD",
            "hybrid_search.sparse_model": "QDRANT_SPARSE_MODEL",'''

content = re.sub(pattern, replacement, content)

with open('src/config.py', 'w') as f:
    f.write(content)
