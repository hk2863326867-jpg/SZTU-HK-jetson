import urllib.request
import json
import time

time.sleep(2)

try:
    url = 'http://127.0.0.1:5000/api/dien/recommend'
    data = {
        'user_id': 'USER001',
        'history_items': ['ITEM001', 'ITEM002'],
        'history_categories': ['CAT001', 'CAT002'],
        'candidate_items': ['ITEM003', 'ITEM004', 'ITEM005'],
        'candidate_categories': ['CAT001', 'CAT002', 'CAT003'],
        'edge_ip': '127.0.0.1'
    }
    
    data_bytes = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=data_bytes, headers={'Content-Type': 'application/json'})
    response = urllib.request.urlopen(req)
    print(response.read().decode())
except Exception as e:
    print(f"Error: {e}")
