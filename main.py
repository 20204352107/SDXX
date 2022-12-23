import io
import math, json
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

#读取数据集标注
train_json = pd.read_json('train.json')
train_json['filename'] = train_json['annotations'].apply(lambda x: x['filename'].replace('\\', '/'))
train_json['period'] = train_json['annotations'].apply(lambda x: x['period'])
train_json['weather'] = train_json['annotations'].apply(lambda x: x['weather'])

train_json.head()

# 用factorize将标签进行编码，这里需要记住编码的次序。
train_json['period'], period_dict = pd.factorize(train_json['period'])
train_json['weather'], weather_dict = pd.factorize(train_json['weather'])
#将period和weather进行编码便于后期的分类

#统计标签
train_json['period'].value_counts()
train_json['weather'].value_counts()


# 自定义数据集
class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df

        # WeatherDataSet继承自DateSet 然后df=传过来的参数
        # 创建一个可调用的Compose对象，它将依次调用每个给定的 transforms
        self.transform = T.Compose([
            T.Resize(size=(340, 340)),
            T.RandomCrop(size=(256, 256)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

    # 设置增加图像的方法
    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        img = Image.open(file_name)
        img = self.transform(img)
        return img, \
               paddle.to_tensor(self.df['period'].iloc[index]), \
               paddle.to_tensor(self.df['weather'].iloc[index])

    def __len__(self):
        return len(self.df)

#加载数据集和验证集
#训练集
#留500张进行验证
#参数1传入的数据集，参数二每一批有多少样本,在每个epoch开始的时候，对数据进行重新排序
train_dataset = WeatherDataset(train_json.iloc[:-500])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 验证集
val_dataset = WeatherDataset(train_json.iloc[-500:])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

from paddle.vision.models import resnet18

#选择resnet18模型
class WeatherModel(paddle.nn.Layer):
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True)#加载CNN网络模型
        backbone.fc = paddle.nn.Identity()#定义全连接层
        self.backbone = backbone

        # 分类1
        self.fc1 = paddle.nn.Linear(512, 4)#对于时间的分类

        # 分类2
        self.fc2 = paddle.nn.Linear(512, 3)#对于天气的分类

    #定义前向激活函数
    def forward(self, x):
        out = self.backbone(x)

        # 同时完成类别1 和 类别2 分类
        log1 = self.fc1(out)
        log2 = self.fc2(out)
        return log1, log2

model = WeatherModel()
model(paddle.to_tensor(np.random.rand(10, 3, 256, 256).astype(np.float32)))

# 定义损失函数和优化器
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001)
criterion = paddle.nn.CrossEntropyLoss()

for epoch in range(0, 30):
    train_loss, val_loss = [], []
    train_Acc1, train_Acc2 = [], []
    val_Acc1, val_Acc2 = [], []

    # 模型训练
    model.train()
    for i, (x, y1, y2) in enumerate(train_loader):
        pred1, pred2 = model(x)

        # 类别1的loss + 类别2的loss为总共的loss
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        train_Acc1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        train_Acc2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

    # 模型验证
    model.eval()
    for i, (x, y1, y2) in enumerate(val_loader):
        pred1, pred2 = model(x)
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        val_loss.append(loss.item())
        val_Acc1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        val_Acc2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

    if epoch % 1 == 0:
        print(f'Loss {np.mean(train_loss):3.5f}/{np.mean(val_loss):3.5f}')
        print(f'period.ACC {np.mean(train_Acc1):3.5f}/{np.mean(val_Acc1):3.5f}')
        print(f'weather.ACC {np.mean(train_Acc2):3.5f}/{np.mean(val_Acc2):3.5f}')

import glob

#测试集数据路径
Test_df = pd.DataFrame({'filename': glob.glob('./test_images/*.jpg')})
Test_df['period'] = 0
Test_df['weather'] = 0
Test_df = Test_df.sort_values(by='filename')
Test_dataset = WeatherDataset(Test_df)
Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=False)
model.eval()
period_pred = []
weather_pred = []

#测试集进行预测
for i, (x, y1, y2) in enumerate(Test_loader):
    pred1, pred2 = model(x)
    period_pred += period_dict[pred1.argmax(1).numpy()].tolist()
    weather_pred += weather_dict[pred2.argmax(1).numpy()].tolist()

Test_df['period'] = period_pred
Test_df['weather'] = weather_pred
submit_json = {
    'annotations':[]
}

#生成测试集结果
for row in Test_df.iterrows():
    submit_json['annotations'].append({
        'filename': 'test_images\\' + row[1].filename.split('/')[-1],
        'period': row[1].period,
        'weather': row[1].weather,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)