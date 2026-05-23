import tensorflow as tf
import os

# 加载模型元数据，查看张量名称
def inspect_model():
    model_path = r"D:\dien_checkpoints\best\ckpt_noshuffDIEN3_d64"
    
    # 禁用TF2行为
    tf.compat.v1.disable_v2_behavior()
    
    # 创建会话
    sess = tf.compat.v1.Session()
    
    # 导入模型元数据
    saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
    
    # 获取默认图
    graph = tf.compat.v1.get_default_graph()
    
    # 查看所有占位符
    print("所有占位符:")
    for op in graph.get_operations():
        if op.type == 'Placeholder':
            print(f"  {op.name}")
    
    # 查看所有变量
    print("\n所有变量:")
    for var in tf.compat.v1.global_variables():
        print(f"  {var.name}")
    
    # 关闭会话
    sess.close()

if __name__ == "__main__":
    inspect_model()