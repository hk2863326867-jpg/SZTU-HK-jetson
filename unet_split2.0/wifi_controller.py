#!/usr/bin/env python3
# WiFi管理控制器 - 封装为类供主服务器调用

import os
import re
import threading

class WiFiController:
    def __init__(self):
        self.HOTSPOT_NAME = "Board_Hotspot"
        self.INTERFACE = "wlP1p1s0"
    
    def get_real_tx_power(self):
        """获取实际传输功率"""
        try:
            res = os.popen(f"iw dev {self.INTERFACE} get txpower").read().strip()
            power_dbm = re.search(r'(\d+\.\d+) dBm', res)
            if power_dbm:
                return str(int(float(power_dbm.group(1))))
            res = os.popen(f"iw dev {self.INTERFACE} info | grep txpower").read().strip()
            power_dbm = re.search(r'txpower (\d+)\.\d+ dBm', res)
            if power_dbm:
                return power_dbm.group(1)
        except Exception as e:
            print(f"读取功率失败：{e}")
        return "20"
    
    def get_real_channel(self):
        """获取实际信道"""
        try:
            res = os.popen(f"iw dev {self.INTERFACE} info | grep channel").read().strip()
            channel = re.search(r'channel (\d+)', res)
            if channel:
                return channel.group(1)
        except Exception as e:
            print(f"读取信道失败：{e}")
        return "6"
    
    def get_connected_devices(self):
        """获取连接设备数"""
        try:
            return str(os.popen(f"iw dev {self.INTERFACE} station dump").read().count("Station"))
        except Exception as e:
            print(f"获取连接设备失败: {e}")
            return "0"
    
    def set_channel_async(self, new_channel):
        """异步设置信道"""
        try:
            current_channel = self.get_real_channel()
            if new_channel != current_channel:
                print(f"信道从{current_channel}修改为{new_channel}，重启热点生效...")
                os.system(f'sudo nmcli connection modify "{self.HOTSPOT_NAME}" 802-11-wireless.channel {new_channel}')
                os.system(f'sudo nmcli connection down "{self.HOTSPOT_NAME}"')
                os.system(f'sudo nmcli connection up "{self.HOTSPOT_NAME}"')
            else:
                print("信道未修改，无需重启")
        except Exception as e:
            print(f"信道设置失败: {e}")
    
    def set_power_async(self, new_tx_power):
        """异步设置功率"""
        try:
            print(f"开始设置功率为{new_tx_power}dBm")
            
            # 方法1：使用iw命令设置功率
            cmd = f'sudo iw dev {self.INTERFACE} set txpower fixed {int(new_tx_power)*100}'
            print(f"执行命令: {cmd}")
            result = os.system(cmd)
            print(f"命令执行结果: {result}")
            
            # 方法2：尝试使用nmcli设置功率（如果iw命令失败）
            if result != 0:
                print("iw命令失败，尝试使用nmcli设置功率")
                cmd2 = f'sudo nmcli connection modify "{self.HOTSPOT_NAME}" 802-11-wireless.tx-power {new_tx_power}'
                print(f"执行命令: {cmd2}")
                result2 = os.system(cmd2)
                print(f"nmcli命令执行结果: {result2}")
                
                # 重启热点使设置生效
                if result2 == 0:
                    print("重启热点使功率设置生效...")
                    os.system(f'sudo nmcli connection down "{self.HOTSPOT_NAME}"')
                    os.system(f'sudo nmcli connection up "{self.HOTSPOT_NAME}"')
            
            print(f"功率设置完成，设置值: {new_tx_power}dBm")
            
        except Exception as e:
            print(f"功率设置失败: {e}")
