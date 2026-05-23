import tensorflow as tf
# 禁用eager execution，使用TensorFlow 1.x风格的API
tf.compat.v1.disable_eager_execution()
import os
import pickle

class DIENEdgeEncoder:
    def __init__(self, model_path, vocab_path):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.load_vocabs()
        self.build_model()
        self.initialize_variables()
    
    def load_vocabs(self):
        """加载词表"""
        try:
            with open(os.path.join(self.vocab_path, 'uid_voc.pkl'), 'rb') as f:
                self.uid_voc = pickle.load(f)
            with open(os.path.join(self.vocab_path, 'mid_voc.pkl'), 'rb') as f:
                self.mid_voc = pickle.load(f)
            with open(os.path.join(self.vocab_path, 'cat_voc.pkl'), 'rb') as f:
                self.cat_voc = pickle.load(f)
            
            self.n_uid = len(self.uid_voc) + 1
            self.n_mid = len(self.mid_voc) + 1
            self.n_cat = len(self.cat_voc) + 1
            print("词表加载成功")
        except Exception as e:
            print(f"词表加载失败: {e}")
            # 如果词表加载失败，使用默认值
            self.uid_voc = {}
            self.mid_voc = {}
            self.cat_voc = {}
            self.n_uid = 48
            self.n_mid = 30000
            self.n_cat = 1000
    
    def build_model(self):
        """构建边缘端编码器模型"""
        # 超参数
        self.EMBEDDING_DIM = 64
        self.LATENT_DIM = 64
        self.NOISE_STD = 0.05
        
        # 输入占位符
        self.mid_his_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
        self.cat_his_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
        self.uid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='uid_batch_ph')
        self.mid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='mid_batch_ph')
        self.cat_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='cat_batch_ph')
        self.mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='mask')
        self.seq_len_ph = tf.compat.v1.placeholder(tf.int32, [None], name='seq_len_ph')
        
        # 嵌入层
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.compat.v1.get_variable("uid_embedding_var", [self.n_uid, self.EMBEDDING_DIM])
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.compat.v1.get_variable("mid_embedding_var", [self.n_mid, self.EMBEDDING_DIM])
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cat_embeddings_var = tf.compat.v1.get_variable("cat_embedding_var", [self.n_cat, self.EMBEDDING_DIM])
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)

        # 特征组合
        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        
        # 计算边缘端的历史表示
        edge_history_repr = self.item_his_eb_sum
        
        # 编码器：生成隐私保护的隐向量
        with tf.compat.v1.variable_scope('DIEN_encoder'):
            # 使用tf.compat.v1.layers.dense确保与TensorFlow 1.x兼容
            edge_hidden = tf.compat.v1.layers.dense(edge_history_repr, 128, activation=tf.nn.relu, name='edge_hidden')
            base_vector = tf.compat.v1.layers.dense(edge_hidden, self.LATENT_DIM, activation=tf.nn.tanh, name='base_vector')
            noise = tf.compat.v1.random_normal(tf.shape(base_vector), mean=0.0, stddev=self.NOISE_STD, name='gaussian_noise')
            noisy_vector = base_vector + noise
            # 固定随机投影 + 非线性压缩，降低从向量反推原始行为序列的可能性
            random_projection = tf.compat.v1.get_variable(
                'random_projection',
                shape=[self.LATENT_DIM, self.LATENT_DIM],
                initializer=tf.compat.v1.random_normal_initializer(stddev=0.2),
                trainable=False
            )
            self.privacy_vector = tf.nn.tanh(tf.matmul(noisy_vector, random_projection), name='protected_vector')
    
    def initialize_variables(self):
        """初始化变量并加载预训练权重"""
        try:
            self.sess = tf.compat.v1.Session()
            
            # 首先初始化所有变量
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
            # 尝试加载预训练权重
            try:
                saver = tf.compat.v1.train.Saver()
                saver.restore(self.sess, self.model_path)
                print(f"预训练权重加载成功: {self.model_path}")
            except Exception as e:
                print(f"预训练权重加载失败，使用随机初始化: {e}")
            
            print("编码器变量初始化成功")
        except Exception as e:
            print(f"编码器初始化失败: {e}")
            raise
    
    def encode(self, user_id, item_id, category_id, item_history, category_history, seq_length):
        """编码用户历史行为为隐向量"""
        # 转换为ID
        uid = self.uid_voc.get(user_id, 0)
        mid = self.mid_voc.get(item_id, 0)
        cat = self.cat_voc.get(category_id, 0)
        mid_his = [self.mid_voc.get(item, 0) for item in item_history]
        cat_his = [self.cat_voc.get(cat, 0) for cat in category_history]
        
        # 填充序列
        max_len = len(mid_his)
        mask = [1.0] * len(mid_his)
        
        # 执行编码
        feed_dict = {
            self.uid_batch_ph: [uid],
            self.mid_batch_ph: [mid],
            self.cat_batch_ph: [cat],
            self.mid_his_batch_ph: [mid_his],
            self.cat_his_batch_ph: [cat_his],
            self.mask: [mask],
            self.seq_len_ph: [seq_length]
        }
        
        privacy_vector = self.sess.run(self.privacy_vector, feed_dict=feed_dict)
        # 确保返回的是Python列表
        return privacy_vector[0].tolist()
    
    def close(self):
        """关闭会话"""
        if hasattr(self, 'sess') and self.sess:
            self.sess.close()
