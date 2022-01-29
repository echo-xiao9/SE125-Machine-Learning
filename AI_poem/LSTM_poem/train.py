from poem import PoemModel
import numpy as np
import torch as t


def train():
    # 获取数据
    raw_data = np.load("tang.npz", allow_pickle=True)
    data = raw_data['data']
    ix2word = raw_data['ix2word'].item()
    word2ix = raw_data['word2ix'].item()
    data = t.from_numpy(data)

    model = PoemModel(len(word2ix), ix2word, word2ix)
    model.train(data, output_path='result.txt')


if __name__ == '__main__':
    train()
