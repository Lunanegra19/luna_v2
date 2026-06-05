"""Test graphify stats API on port 8080"""
import urllib.request, json

try:
    r = urllib.request.urlopen('http://127.0.0.1:8080/api/graphify/stats')
    data = json.loads(r.read())
    print(f"status: {data.get('status')}")
    print(f"total_nodes: {data.get('total_nodes')}")
    print(f"total_links: {data.get('total_links')}")
    print(f"total_communities: {data.get('total_communities')}")
    print(f"file_types: {data.get('file_types')}")
    print(f"density: {data.get('density')}")
    print("SUCCESS: graphify/stats API respondiendo correctamente")
except Exception as e:
    print(f"ERROR: {e}")
