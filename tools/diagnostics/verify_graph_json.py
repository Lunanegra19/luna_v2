"""Verify graph.json node and link counts"""
import json

with open('/root/luna_v2/graphify/out/graph.json', 'r') as f:
    d = json.load(f)

nodes = d.get('nodes', [])
links = d.get('links', [])
print(f"Nodes: {len(nodes)}")
print(f"Links: {len(links)}")

# Check community field
communities = set()
file_types = {}
for n in nodes:
    c = n.get('community')
    if c is not None:
        communities.add(c)
    ft = n.get('file_type', n.get('type', 'unknown'))
    file_types[ft] = file_types.get(ft, 0) + 1

print(f"Communities: {len(communities)}")
print(f"File types: {file_types}")
