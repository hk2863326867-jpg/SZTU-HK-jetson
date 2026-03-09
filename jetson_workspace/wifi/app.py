from flask import Flask, render_template, request, redirect, jsonify
import os
import threading
import re  # 正则解析功率

app = Flask(__name__)
HOTSPOT_NAME = "Board_Hotspot"  # 你的热点名称
INTERFACE = "wlP1p1s0"          # 你的无线接口名

# ---------------------- 工具函数 ----------------------
# 精准读取当前功率
def get_real_tx_power():
    try:
        res = os.popen(f"iw dev {INTERFACE} get txpower").read().strip()
        power_dbm = re.search(r'(\d+\.\d+) dBm', res)
        if power_dbm:
            return str(int(float(power_dbm.group(1))))
        res = os.popen(f"iw dev {INTERFACE} info | grep txpower").read().strip()
        power_dbm = re.search(r'txpower (\d+)\.\d+ dBm', res)
        if power_dbm:
            return power_dbm.group(1)
    except Exception as e:
        print(f"读取功率失败：{e}")
    return "20"

# 读取当前实际信道（精准）
def get_real_channel():
    try:
        res = os.popen(f"iw dev {INTERFACE} info | grep channel").read().strip()
        channel = re.search(r'channel (\d+)', res)
        if channel:
            return channel.group(1)
    except Exception as e:
        print(f"读取信道失败：{e}")
    return "6"

# 读取所有基础参数
def get_wifi_params():
    return {
        "ssid": HOTSPOT_NAME,
        "interface": INTERFACE,
        "channel": get_real_channel(),
        "tx_power_actual": get_real_tx_power(),
        "mode": "AP"
    }

# ---------------------- 独立修改逻辑 ----------------------
# 仅修改信道（按需重启）
def set_channel_async(new_channel):
    current_channel = get_real_channel()
    if new_channel != current_channel:
        print(f"信道从{current_channel}修改为{new_channel}，重启热点生效...")
        os.system(f'sudo nmcli connection modify "{HOTSPOT_NAME}" 802-11-wireless.channel {new_channel}')
        os.system(f'sudo nmcli connection down "{HOTSPOT_NAME}"')
        os.system(f'sudo nmcli connection up "{HOTSPOT_NAME}"')
    else:
        print("信道未修改，无需重启")

# 仅修改功率（不重启）
def set_power_async(new_tx_power):
    print(f"修改功率为{new_tx_power}dBm，实时生效（硬件可能自动修正）")
    os.system(f'sudo iw dev {INTERFACE} set txpower fixed {int(new_tx_power)*100}')

# ---------------------- API接口 ----------------------
@app.route('/get_real_power')
def get_real_power():
    return jsonify({"power": get_real_tx_power()})

@app.route('/get_real_channel')
def get_real_channel_api():
    return jsonify({"channel": get_real_channel()})

@app.route('/get_connected_devices')
def get_connected_devices_api():
    try:
        return str(os.popen(f"iw dev {INTERFACE} station dump").read().count("Station"))
    except:
        return "0"

# ---------------------- 页面路由 ----------------------
@app.route('/')
def index():
    wifi_params = get_wifi_params()
    connected_devices = get_connected_devices_api()
    return render_template('index.html',
                           wifi_params=wifi_params,
                           connected_devices=connected_devices)

# 独立处理信道修改
@app.route('/set_channel', methods=['POST'])
def set_channel():
    new_channel = request.form.get("channel", get_real_channel())
    threading.Thread(target=set_channel_async, args=(new_channel,)).start()
    feedback = f"✅ 信道修改提交成功！<br>"
    if new_channel != get_real_channel():
        feedback += f"- 信道将从{get_real_channel()}修改为{new_channel}，已重启热点生效<br>"
    else:
        feedback += f"- 信道未修改，无需重启<br>"
    feedback += f"<a href='/'>返回配置面板</a>"
    return feedback

# 独立处理功率修改
@app.route('/set_power', methods=['POST'])
def set_power():
    new_tx_power = request.form.get("tx_power", get_real_tx_power())
    threading.Thread(target=set_power_async, args=(new_tx_power,)).start()
    feedback = f"✅ 功率修改提交成功！<br>"
    feedback += f"- 功率设置为{new_tx_power}dBm（实时生效，硬件可能自动修正）<br>"
    feedback += f"<a href='/'>返回配置面板</a>"
    return feedback

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)