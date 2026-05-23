import tensorflow as tf
# 禁用eager execution，使用TensorFlow 1.x风格的API
tf.compat.v1.disable_eager_execution()
import os
import pickle
from tf1_compat import dense as tf1_dense

class DIENLocalDecoder:
    def __init__(self, model_path, vocab_path):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.load_vocabs()
        self.build_model()
        self.load_weights()
    
    def load_vocabs(self):
        """加载词表"""
        with open(os.path.join(self.vocab_path, 'uid_voc.pkl'), 'rb') as f:
            self.uid_voc = pickle.load(f)
        with open(os.path.join(self.vocab_path, 'mid_voc.pkl'), 'rb') as f:
            self.mid_voc = pickle.load(f)
        with open(os.path.join(self.vocab_path, 'cat_voc.pkl'), 'rb') as f:
            self.cat_voc = pickle.load(f)
        
        # 硬编码词表大小，与预训练权重匹配
        self.n_uid = 48
        self.n_mid = 367983
        self.n_cat = 113
    
    def build_model(self):
        """构建本地端解码器模型"""
        # 超参数
        self.EMBEDDING_DIM = 64
        self.LATENT_DIM = 64
        
        # 输入占位符
        self.uid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='uid_batch_ph')
        self.mid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='mid_batch_ph')
        self.cat_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='cat_batch_ph')
        self.privacy_vector_ph = tf.compat.v1.placeholder(tf.float32, [None, self.LATENT_DIM], name='privacy_vector_ph')
        self.target_ph = tf.compat.v1.placeholder(tf.float32, [None, None], name='target_ph')
        self.lr = tf.compat.v1.placeholder(tf.float64, [])
        
        # 嵌入层
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.compat.v1.get_variable("uid_embedding_var", [self.n_uid, self.EMBEDDING_DIM])
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.compat.v1.get_variable("mid_embedding_var", [self.n_mid, self.EMBEDDING_DIM])
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)

            self.cat_embeddings_var = tf.compat.v1.get_variable("cat_embedding_var", [self.n_cat, self.EMBEDDING_DIM])
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)

        # 特征组合
        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        
        # 解码器：使用隐向量进行预测
        self.y_hat = self.dien_decoder(self.privacy_vector_ph)
    
    def dien_decoder(self, protected_vector):
        """本地端解码器：不可逆隐向量 + 当前候选项 -> 点击概率"""
        with tf.compat.v1.variable_scope('DIEN_decoder'):
            item_proj = tf1_dense(self.item_eb, self.LATENT_DIM, activation=None, name='item_proj')
            fusion = tf.concat(
                [
                    self.uid_batch_embedded,
                    self.item_eb,
                    protected_vector,
                    item_proj * protected_vector
                ],
                1
            )
            return self.build_fcn_net(fusion, use_dice=True)
    
    def build_fcn_net(self, inp, use_dice = False):
        # 简化实现，使用基本的全连接层和ReLU激活函数
        dnn1 = tf1_dense(inp, 200, activation=tf.nn.relu, name='f1')
        dnn2 = tf1_dense(dnn1, 80, activation=tf.nn.relu, name='f2')
        dnn3 = tf1_dense(dnn2, 2, activation=None, name='f3')
        y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.math.log(y_hat) * self.target_ph)
            self.loss = ctr_loss
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), self.target_ph), tf.float32))

        return y_hat
    
    def load_weights(self):
        """加载预训练权重"""
        try:
            self.sess = tf.compat.v1.Session()
            
            # 获取所有可训练变量
            all_vars = tf.compat.v1.global_variables()
            
            # 过滤出我们需要的变量（排除优化器相关的变量）
            var_list = []
            for var in all_vars:
                var_name = var.name
                # 排除优化器相关的变量
                if 'Adam' not in var_name and 'beta1' not in var_name and 'beta2' not in var_name:
                    var_list.append(var)
            
            # 打印所有要加载的变量名称，以便调试
            print("要加载的变量:")
            for var in var_list:
                print(f"  - {var.name}")
            
            # 创建Saver对象，只加载指定的变量
            saver = tf.compat.v1.train.Saver(var_list=var_list)
            
            saver.restore(self.sess, self.model_path)
            print("解码器权重加载成功")
        except Exception as e:
            print(f"解码器加载失败: {e}")
            # 尝试使用更简单的方法，只加载嵌入层变量
            try:
                print("尝试只加载嵌入层变量...")
                # 只加载嵌入层变量
                embedding_vars = []
                for var in all_vars:
                    var_name = var.name
                    if 'embedding_var' in var_name:
                        embedding_vars.append(var)
                
                print("要加载的嵌入层变量:")
                for var in embedding_vars:
                    print(f"  - {var.name}")
                
                if embedding_vars:
                    saver = tf.compat.v1.train.Saver(var_list=embedding_vars)
                    saver.restore(self.sess, self.model_path)
                    print("嵌入层变量加载成功")
                else:
                    print("没有找到嵌入层变量")
            except Exception as e2:
                print(f"嵌入层变量加载失败: {e2}")
            raise
    
    def decode(self, user_id, item_id, category_id, privacy_vector):
        """解码隐向量，生成推荐结果"""
        # 转换为ID
        uid = self.uid_voc.get(user_id, 0)
        mid = self.mid_voc.get(item_id, 0)
        cat = self.cat_voc.get(category_id, 0)
        
        # 执行解码
        feed_dict = {
            self.uid_batch_ph: [uid],
            self.mid_batch_ph: [mid],
            self.cat_batch_ph: [cat],
            self.privacy_vector_ph: privacy_vector
        }
        
        prediction = self.sess.run(self.y_hat, feed_dict=feed_dict)
        return prediction
    
    def close(self):
        """关闭会话"""
        if hasattr(self, 'sess') and self.sess:
            self.sess.close()
