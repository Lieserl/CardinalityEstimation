### 准备工作

#### 需要的包

1. 在该项目的根目录下，在 *Pycharm* 内置的终端中输入
```angular2html
pip install torch
pip install pytest
pip install matplotlib
```
注：请确保你已为该项目添加 *Python* 解释器，如果仍安装失败，可以将 *pip* 替换为 *pip3*

#### Cuda 的安装

1. 在 *Nvidia* 官网下载 *Cuda* 驱动程序，你可以直接在 *Bing* 中搜索 *cuda download*
2. 如果安装失败，查看当时页面中提示安装失败的项，重新安装时选择自定义安装，选择不安装之前安装失败的那一项

#### Pytorch Cuda 支持

1. 进入 *Pytorch* 官网，往下翻能看到 *install pytorch*，依次选择 *Stable*，*Windows*，*Pip*，*Python*，*CUDA 12.1*
2. *Pytorch* 会在选择完后紧接着提供对应的安装指令
3. 然后打开 *Pycharm*，在该项目的根目录下，在内置的终端中输入提供的指令，安装完成

#### 测试

安装完成后可以在项目中使用
```angular2html
print(torch.cuda.is_available()) 
```
确定 *Cuda* 的支持状态

### 模型调参

你可以调节的超参数分为两部分：
1. 学习率，优化器，
2. 模型参数

#### 学习率

可以将其理解为采样率。
1. 过高会欠拟合（对应采样率过低），可能错过当前最优答案，优点是可以在少次 *epoch* 中观察模型的初步收敛效果（虽然在本项目中没啥用）
2. 过低会过拟合（对应采样率过高），可能会导致模型生成的是对应训练集数据的特解。
3. 在 1e-5 ~ 1e-1 之内调

#### 优化器

1. 目前使用的优化器为 *Adam*，你也可以尝试 *SGD* 等优化器
2. *SGD* 的有效范围远小于 *Adam*，但最终效果比 *Adam* 好

#### 模型参数

1. 每层线性层的超参数，你可以在 *learn_from_query* 的 *est_ai1* 中找到
```angular2html
model1 = model.Model1(num_input=15, para_1=100, para_2=50, num_output=1)
```
你可以修改的是 *para_1* 和 *para_2*

2. 非线性激活函数，你可以在 *model* 中找到，可以查看官方文档进行调试或换用其他非线性激活函数
```angular2html
self.model = nn.Sequential(
    nn.Linear(num_input, para_1),
    nn.ReLU(),
    nn.Linear(para_1, para_2),
    nn.ReLU(),
    nn.Linear(para_2, num_output),
)
```
其中 *nn.ReLU()* 为非线性激活函数

3. 隐藏层层数，可以在 *model.py* 中修改，具体操作示例如下 
```angular2html
def __init__(self, num_input, para_1, para_2, num_output):
    super(Model1, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(num_input, para_1),
        nn.ReLU(),
        nn.Linear(para_1, para_2),
        nn.ReLU(),
        nn.Linear(para_2, num_output),
    )
```
将 
```angular2html
nn.Linear(para_1, para_2),
nn.ReLU(),
```
复制并添加到最后一个 *nn.Linear* 上方，并新增 *para_3* 变量，如下所示
```angular2html
def __init__(self, num_input, para_1, para_2, para_3, num_output):
    super(Model1, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(num_input, para_1),
        nn.ReLU(),
        nn.Linear(para_1, para_2),
        nn.ReLU(),
        nn.Linear(para_2, para_3),
        nn.ReLU(),
        nn.Linear(para_3, num_output),
    )
```
注意：上一层线性层的输出要和下一层线性层的输入一致
接着修改 *learn_from_query* 中模型初始化语句
```angular2html
# Before
model1 = model.Model1(num_input=15, para_1=100, para_2=50, num_output=1)

# After
model1 = model.Model1(num_input=15, para_1=100, para_2=50, para_3 = ?, num_output=1)
```
