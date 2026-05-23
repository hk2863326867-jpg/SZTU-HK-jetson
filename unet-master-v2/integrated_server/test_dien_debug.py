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
    print(f"Sending request: {json.dumps(data)}")
    
    req = urllib.request.Request(url, data=data_bytes, headers={'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(req, timeout=10)
        print(f"Response code: {response.getcode()}")
        result = response.read().decode()
        print(f"Response: {result}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(f"Error body: {e.read().decode()}")
    except urllib.error.URLError as e:
        print(f"URL Error: {e}")
        
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
