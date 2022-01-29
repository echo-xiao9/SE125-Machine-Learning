## README

### 运行说明

项目文件中config.py定义了模型中的一些参数设置，model.py与poem.py定义了模型类以及模型的一些接口。调用模型进行训练部分的代码位于文件train.py中，调用模型进行输出测试部分的代码位于文件test.py中。

因此，若需要训练模型，可以在config中进行参数设置，并运行train.py。若需要对模型进行测试，可以在config中设置加载的模型路径，并运行test.py
