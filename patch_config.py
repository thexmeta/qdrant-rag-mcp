import json

with open('config/server_config.json', 'r') as f:
    config = json.load(f)

config['hybrid_search']['sparse_method'] = "${QDRANT_SPARSE_METHOD:-bm42}"
config['hybrid_search']['sparse_model'] = "${QDRANT_SPARSE_MODEL:-Qdrant/bm42-all-minilm-l6-v2-attentions}"

with open('config/server_config.json', 'w') as f:
    json.dump(config, f, indent=2)

with open('src/config.py', 'r') as f:
    content = f.read()

content = content.replace('"exact_match_bonus": 0.2', '"exact_match_bonus": 0.2,\n        "sparse_method": "bm42",\n        "sparse_model": "Qdrant/bm42-all-minilm-l6-v2-attentions"')

with open('src/config.py', 'w') as f:
    f.write(content)
