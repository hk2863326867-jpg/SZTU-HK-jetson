import urllib.request
import json

# 测试DIEN推荐API
url = 'http://127.0.0.1:5000/api/dien/recommend'
data = {
    'user_id': 'USER001',
    'history_items': ['ITEM001', 'ITEM002'],
    'history_categories': ['CAT001', 'CAT002'],
    'candidate_items': ['ITEM003', 'ITEM004', 'ITEM005'],
    'candidate_categories': ['CAT001', 'CAT002', 'CAT003']
}

data_bytes = json.dumps(data).encode('utf-8')
req = urllib.request.Request(url, data=data_bytes, headers={'Content-Type': 'application/json'})

try:
    response = urllib.request.urlopen(req, timeout=10)
    print("DIEN recommendation response:", response.read().decode())
except Exception as e:
    print("DIEN recommendation error:", e)
