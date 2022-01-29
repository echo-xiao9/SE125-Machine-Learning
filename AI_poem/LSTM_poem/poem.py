from config import *
from model import PoetryModel
from torch import optim
import torch.nn as nn
import torch as t
from torchnet import meter
import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class PoemModel:
    # 初始化
    def __init__(self, vocab_size: int, ix2word, word2ix, try_use_train_model=False):
        # 根据Config，初始化诗歌生成模型的基本参数
        self.model = PoetryModel(vocab_size, Config.embedding_dim, Config.hidden_dim)
        self.param_updater = optim.Adam(self.model.parameters(), lr=Config.lr)
        self.criterion = nn.CrossEntropyLoss()

        # 选择训练后的模型
        if try_use_train_model and Path(Config.latest_model_path).is_file():
            self.model.load_state_dict(t.load(Config.latest_model_path, 'cpu'))

        # 移动模型至对应设备
        if Config.use_gpu:
            self.device = t.device("cuda")
        else:
            self.device = t.device("cpu")
        self.model.to(self.device)

        # 记录loss
        self.loss_meter = meter.AverageValueMeter()

        # 记录ix2word与word2ix
        self.ix2word = ix2word
        self.word2ix = word2ix

    # 给定首句生成诗歌
    def generate(self, start_words, prefix_words=None):
        results = list(start_words)
        start_words_len = len(start_words)
        # 第一个词语是<START>
        inputs = t.Tensor([self.word2ix['<START>']]).view(1, 1).long()
        if Config.use_gpu:
            inputs = inputs.cuda()
        hidden = None

        # 若有风格前缀，则先用风格前缀生成hidden
        if prefix_words:
            # 第一个input是<START>，后面就是prefix中的汉字
            # 第一个hidden是None，后面就是前面生成的hidden
            for word in prefix_words:
                _, hidden = self.model(inputs, hidden)
                inputs = inputs.data.new([self.word2ix[word]]).view(1, 1)

        # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
        # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
        for i in range(Config.max_gen_len):
            output, hidden = self.model(inputs, hidden)
            # print(output.shape)
            # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
            # 最后的hidden
            if i < start_words_len:
                w = results[i]
                inputs = inputs.data.new([self.word2ix[w]]).view(1, 1)
            # 否则将output作为下一个input进行
            else:
                # print(output.data[0].topk(1))
                top_index = output.data[0].topk(1)[1][0].item()
                w = self.ix2word[top_index]
                results.append(w)
                inputs = inputs.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break
        return results

    # 生成藏头诗
    def gen_acrostic(self, start_words, prefix_words=None):
        result = []
        start_words_len = len(start_words)
        inputs = (t.Tensor([self.word2ix['<START>']]).view(1, 1).long())
        if Config.use_gpu:
            inputs = inputs.cuda()

        # 指示已经生成了几句藏头诗
        index = 0
        pre_word = '<START>'
        hidden = None

        # 存在风格前缀，则生成hidden
        if prefix_words:
            for word in prefix_words:
                output, hidden = self.model(inputs, hidden)
                inputs = (inputs.data.new([self.word2ix[word]])).view(1, 1)

        # 开始生成诗句
        for i in range(Config.max_gen_len):
            output, hidden = self.model(inputs, hidden)
            top_index = output.data[0].topk(1)[1][0].item()
            w = self.ix2word[top_index]

            # 如果上个字是句末
            if pre_word in {'。', '，', '?', '！', '<START>'}:
                if index == start_words_len:
                    break
                else:
                    w = start_words[index]
                    index += 1
                    inputs = (inputs.data.new([self.word2ix[w]])).view(1, 1)
            else:
                inputs = (inputs.data.new([top_index])).view(1, 1)
            result.append(w)
            pre_word = w
        return result

    # 训练
    def train(self, data: t.Tensor, output_path: str):
        data_loader = DataLoader(data,
                                 batch_size=Config.batch_size,
                                 shuffle=True,
                                 num_workers=2)

        loss_array = []
        f = open(output_path, 'w', encoding='utf-8')
        for epoch in range(Config.epoch):
            self.loss_meter.reset()
            for li, data_ in tqdm.tqdm(enumerate(data_loader)):

                # 获取数据并转移到设备上
                data_ = data_.long().transpose(1, 0).contiguous()
                data_ = data_.to(self.device)
                self.param_updater.zero_grad()

                # n个句子，前n-1句作为输入，后n-1句作为输出，二者一一对应
                input_, target = data_[:-1, :], data_[1:, :]
                output, _ = self.model(input_)

                # 计算损失并更新参数
                loss = self.criterion(output, target.view(-1))
                loss.backward()
                self.param_updater.step()
                self.loss_meter.add(loss.item())

                # 进行可视化
                if (1 + li) % Config.plot_every == 0:
                    print("训练损失为%s" % (str(self.loss_meter.mean)))
                    f.write("训练损失为%s" % (str(self.loss_meter.mean)))
                    loss_array.append(self.loss_meter.mean)
                    for word in list(u"春江花朝秋月夜"):
                        gen_poetry = ''.join(self.generate(word))
                        print(gen_poetry)
                        f.write(gen_poetry)
                        f.write("\n\n\n")
                        f.flush()
            t.save(self.model.state_dict(), '%s_%s.pth' % (Config.model_prefix, epoch))

        x = range(len(loss_array))
        plt.plot(x, loss_array, label="Loss")
        plt.title("Train Lossy")
        plt.legend()
        plt.show()
