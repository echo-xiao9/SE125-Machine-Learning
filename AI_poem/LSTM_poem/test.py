from poem import PoemModel
import numpy as np


def test():

    # 初始化部分
    print("正在初始化......")
    raw_data = np.load("tang.npz", allow_pickle=True)
    # 加载ix2word与word2ix
    ix2word = raw_data['ix2word'].item()
    word2ix = raw_data['word2ix'].item()
    # 加载模型
    model = PoemModel(len(word2ix), ix2word, word2ix, True)
    print("初始化完成！\n")

    while True:
        print("欢迎使用唐诗生成器，\n"
              "输入1 进入首句生成模式\n"
              "输入2 进入藏头诗生成模式\n")
        mode = int(input())

        if mode == 1:
            print("请输入诗句，以表示您想要的诗歌意境")
            print("可以直接回车跳过")
            prefix_words = str(input())
            try:
                print("请输入您想要的诗歌首句，可以是五言或七言")
                start_words = str(input())

                gen_poetry = ''.join(model.generate(start_words, prefix_words))
                print("生成的诗句如下：%s\n" % gen_poetry)
            except Exception:
                print("您指定的藏头部分包含本诗词库未加载的汉字，请再次输入")

        elif mode == 2:
            print("请输入诗句，以表示您想要的诗歌意境")
            print("可以直接回车跳过")
            prefix_words = str(input())
            try:
                print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
                start_words = str(input())

                gen_poetry = ''.join(model.gen_acrostic(start_words, prefix_words))
                print("生成的诗句如下：%s\n" % gen_poetry)
            except Exception:
                print("您指定的藏头部分包含本诗词库未加载的汉字，请再次输入")


if __name__ == '__main__':
    test()
