# pytorch #
## 数据加载 ##
1. dataset
2. dataloader
###dataset###
data = datasets.CIFAR10("./data/", transform=transform（转换）, train=True（是否用于train）, download=True)
####import torch.utils.data.dataset####
	这是一个类可以通过继承，是一个表示数据集的抽象类。任何自定义的数据集都需要继承这个类并覆写相关方法。
####dataLoader####
	from torch.utils.data import DataLoader
	1. epoch：所有的训练样本输入到模型中称为一个epoch；
	2. iteration：一批样本输入到模型中，成为一个Iteration;
	3. batchszie：批大小，决定一个epoch有多少个Iteration；
	4. 迭代次数（iteration）=样本总数（epoch）/批尺寸（batchszie）
	5. dataset (Dataset) – 决定数据从哪读取或者从何读取；
	6. batch_size (python:int, optional) – 批尺寸(每次训练样本个数,默认为１）
	从datasets取数据
##tenaorboard##
    可视化工具通过网页查看
###torch.utils.tenaorboard###
	from torch.utils.tenaorboard import SummaryWriter
	writer=SummaryWriter('table')
	writer.add_image()
	writer.add_scalar(x,y)
###transforms###
####它是一个类合集可以对数据转换tensor####
	transforms.PILTotensor().方法
	
	transforms.TOTensor()
####注意它有张量，H x W x C####
###resize###
	a=transforms.Resize(x,y)
	b=a(img)
###Normlise(mean,std)###
	方差，均值
	传入Totensor
###Compose###
	a=transforms.Resize(x)
	b=transforms.Totensor()
	c=transforms.Compose([a,b])
###randomCrop###
	b=transforms.Totensor()
	a=transforms.RandomCrop(x,y)
	c=transforms.Compose([b,a])
##Layer##
###卷积层###
####conv1d####
	Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
	in_channels:输入0的个数
	dilation:卷积kernel中元素的间向量特征维度
	out_channels:输入向量经过Conv1d后的特征维度，out_channels等于几，就有几个卷积的kernel.
	kernel_size:卷积核大小
	stride:步长
	padding:输入向量的每一侧填充距
	bias:是否存在bias
####conv2d####
	Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
	in_channels:输入向量特征维度
	out_channels:输入向量经过Conv2d后的特征维度，out_channels等于几，就有几个卷积的kernel.
	kernel_size:卷积核大小
	stride:步长
	padding:输入向量的每一侧填充0的个数
	dilation:卷积kernel中元素的间距
	bias:是否存在bias
	【注】与Conv1d不同的是，kernel_size, stride, padding, dilation可以是a single int(高=宽) 或 a tuple of two ints (高=第一个数字， 宽=第二个数字)
###池化层###
####MaxPool1d####
	kernel_size：滑动窗口的大小，即pool的大小
	stride：滑动窗口的步长，默认为kernel_size
	padding: 填充，默认无穷大
	dilation：滑动窗口中元素的间隔
	return_indices：是否返回max值的索引
####MaxPool2d####
	kernel_size：池化窗口大小，具有长宽两个方向
	stride：池化窗口步长，默认为kernel_size
	padding: 填充
	dilation: 池化窗口中元素的间隔
	return_indices：是否返回max值的索引
	【注】与MaxPool1d不同的是，kernel_size, stride, padding, dilation可以是a single int(高=宽) 或 	a tuple of two ints (高=第一个数字， 宽=第二个数字)
###激活函数###
####ReLU####
	Input: (N, *): N表示batch size, *意味着可以是任意维度
	Output: (N, *): 与Input保持一致
####Sigmoid####
####LogSigmoid####
###线性层###
####Linear####
	in_features: 输入样本的特征维度
	out_features: 输出样本的特征维度
	bias: 是否需要bias参数, 默认为True
#损失函数--只能求单个batch的损失
	L1损失：CLASS torch.nn.L1Loss(size_average=None,reduce=None,reduction='mean')

	reduction="mean",计算平均绝对误差；reduction="sum",计算绝对误差之和

	import torch
	from torch import nn
	loss = nn.L1Loss(reduction="sum")
	input = torch.tensor([[1,2],[3,4]],dtype=torch.float)
	target = torch.tensor([[1,2],[3,5]],dtype=torch.float)
	output = loss(input, target)#loss的输入和输出必须都是float/complex类型
	print(output)


##L2损失：CLASS torch.nn.MSELoss (size_average=None,reduce=None,reduction='mean')

	reduction="mean",计算平均平方误差；reduction="sum",计算平方误差之和。

	import torch
	from torch import nn
	loss = nn.MSELoss(reduction="mean")
	input = torch.tensor([[1,2],[3,4]],dtype=torch.float)
	target = torch.tensor([[1,2],[3,6]],dtype=torch.float)
	output = loss(input, target)
	print(output)

##交叉熵损失：CLASS torch.nn.CrossEntropyLoss(weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean',label_smoothing=0.0)

 Example of target with class indices
	loss = nn.CrossEntropyLoss()

	input = torch.randn(3, 5, requires_grad=True)

	target = torch.empty(3, dtype=torch.long).random_(5)

	output = loss(input, target)

	output.backward()


 Example of target with class probabilities

	input = torch.randn(3, 5, requires_grad=True)

	target = torch.randn(3, 5).softmax(dim=1)

	output = loss(input, target)

	output.backward()

#优化器（执行算法）
	step1: 建立优化器，此处的model.parameter（）要使用模型的实例对象非类名。

	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	step2: 执行优化器，step方法更新参数，一般要进行多轮训练，只需要在外面加一个for循环即可。

	for input, target in dataset:
    	optimizer.zero_grad()#梯度清零
    	output = model(input)
    	loss = loss_fn(output, target)
    	loss.backward()
    	optimizer.step()
#注：SGD算法会随机初始化权重和偏置，可能会得到不同的结果，所以最好设置一个随机数种子torch.manual_seed(seed)，保证每次运行得到相同结果。
