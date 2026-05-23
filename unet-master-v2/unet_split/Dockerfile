FROM nvcr.io/nvidia/pytorch:24.12-py3

# 安装 scikit-image
RUN pip install scikit-image

# 设置工作目录
WORKDIR /workspace

# 暴露端口
EXPOSE 9000

# 默认命令
CMD ["python", "jetson_server.py"]
