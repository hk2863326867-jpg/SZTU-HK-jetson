import urllib.request
import json
import time

def test_dien_recommendation():
    """测试DIEN推荐API"""
    print("=== 测试DIEN推荐系统 ===")
    
    # 测试数据
    url = 'http://127.0.0.1:5000/api/dien/recommend'
    data = {
        'user_id': 'USER001',
        'history_items': ['ITEM001', 'ITEM002'],
        'history_categories': ['CAT001', 'CAT002'],
        'candidate_items': ['ITEM003', 'ITEM004', 'ITEM005'],
        'candidate_categories': ['CAT001', 'CAT002', 'CAT003'],
        'edge_ip': '192.168.55.1'  # Jetson的IP
    }
    
    try:
        data_bytes = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=data_bytes, headers={'Content-Type': 'application/json'})
        
        print("发送请求到:", url)
        print("请求数据:", json.dumps(data, indent=2))
        
        start_time = time.time()
        response = urllib.request.urlopen(req, timeout=30)
        elapsed_time = time.time() - start_time
        
        result = json.loads(response.read().decode())
        
        print(f"\n响应时间: {elapsed_time:.2f}秒")
        print("响应状态:", result.get('status'))
        
        if result.get('status') == 'success':
            print("\n推荐结果:")
            for i, rec in enumerate(result['recommendations']):
                print(f"  {i+1}. 商品ID: {rec['item_id']}, 分类: {rec['category_id']}, 评分: {rec['score']:.4f}")
        else:
            print("\n错误信息:", result.get('error'))
            
    except urllib.error.HTTPError as e:
        print(f"HTTP错误: {e.code}")
        try:
            error_data = json.loads(e.read().decode())
            print("错误详情:", error_data)
        except:
            print("无法解析错误详情")
    except urllib.error.URLError as e:
        print(f"网络错误: {e.reason}")
    except Exception as e:
        print(f"未知错误: {e}")

if __name__ == '__main__':
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(3)
    
    # 运行测试
    test_dien_recommendation()
