import urllib.request
import time

time.sleep(2)
try:
    response = urllib.request.urlopen('http://127.0.0.1:5000/api/health')
    print(response.read().decode())
except Exception as e:
    print(f"Error: {e}")
