

class Config(object):
    num_layers = 3  # LSTM层数
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    lr = 1e-3  # 学习率
    use_gpu = True
    epoch = 50
    batch_size = 16  # 每次迭代样本数量
    max_len = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 200  # 每200个batch可视化一次
    max_gen_len = 200  # 生成诗歌最长长度
    latest_model_path = "checkpoints/WithoutPreTrain.pth"  # 需要加载的模型路径
    model_prefix = 'checkpoints/tang'  # 训练时模型保存的路径前缀
    embedding_dim = 256
    hidden_dim = 512
    test_prefix = u"春江花朝秋月夜"
