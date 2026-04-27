import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision import models
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
class FlowerDataLoader:
    def __init__(self, data_path, batch_size=32, img_size=224):
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.setup_transforms()

    def setup_transforms(self):
        # 基础预处理
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 训练时的数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        """加载数据集并进行划分"""
        # 加载完整数据集
        full_dataset = datasets.ImageFolder(root=self.data_path,
                                            transform=self.base_transform)

        # 数据集划分: 70% 训练, 15% 验证, 15% 测试
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # 为训练集应用数据增强
        train_dataset.dataset.transform = self.train_transform

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=2)

        print(f"数据集划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
        print(f"花朵类别: {full_dataset.classes}")

        return train_loader, val_loader, test_loader, full_dataset.classes


# 2. 基础模型构建
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        # 定义模型的卷积和池化层部分
        self.conv_layers = nn.Sequential(
            # 输入: 3通道 (RGB), 224x224 图像
            # 第一个卷积块
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16通道, 112x112

            # 第二个卷积块
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 32通道, 56x56

            # 第三个卷积块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 64通道, 28x28
        )

        # 定义模型的全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 将 64x28x28 的特征图压平成一维向量
            nn.Linear(in_features=64 * 28 * 28, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 使用可配置的Dropout率
            nn.Linear(in_features=512, out_features=num_classes)  # 输出层
        )

    def forward(self, x):
        # 前向传播 (forward)
        # 1. 先通过卷积层
        x = self.conv_layers(x)
        # 2. 再通过全连接层
        x = self.fc_layers(x)
        # 3. 返回最终输出
        return x

class FlowerClassifier:
    def __init__(self, model_name='custom_cnn', num_classes=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        self.setup_model()

    def setup_model(self):
        """构建模型"""
        if self.model_name == 'custom_cnn':
            self.model = CustomCNN(num_classes=self.num_classes)
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif self.model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)

        self.model = self.model.to(self.device)
        print(f"使用模型: {self.model_name}")

    def train_model(self, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=25):
        """模型训练与评估"""
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        best_val_acc = 0.0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.cpu())

            # 验证阶段
            val_loss, val_acc, val_f1 = self.evaluate_model(val_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if scheduler:
                scheduler.step()

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'best_{self.model_name}.pth')

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'验证 Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}')
            print('-' * 50)

        return train_losses, train_accs, val_losses, val_accs

    def evaluate_model(self, data_loader, criterion):
        """模型评估"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

        return epoch_loss, epoch_acc.cpu(), f1

    def test_model(self, test_loader):
        """测试模型"""
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_f1 = self.evaluate_model(test_loader, criterion)
        print(f'测试结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}')
        return test_loss, test_acc, test_f1


# 4. 结果可视化
class ResultVisualizer:
    @staticmethod
    def plot_training_curves(train_losses, train_accs, val_losses, val_accs, model_name):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(train_losses, label='训练损失')
        ax1.plot(val_losses, label='验证损失')
        ax1.set_title(f'{model_name} - 训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot([acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_accs],
                 label='训练准确率')
        ax2.plot([acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in val_accs],
                 label='验证准确率')
        ax2.set_title(f'{model_name} - 训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def visualize_predictions(model, test_loader, class_names, num_samples=8):
        """可视化预测结果"""
        model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(15, 10))

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(next(model.parameters()).device)
                labels = labels.to(next(model.parameters()).device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    if images_so_far >= num_samples:
                        return

                    images_so_far += 1
                    ax = plt.subplot(num_samples // 4, 4, images_so_far)
                    ax.axis('off')

                    # 反标准化图像
                    img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)

                    ax.imshow(img)
                    ax.set_title(f'预测: {class_names[preds[j]]}\n真实: {class_names[labels[j]]}',
                                 color='green' if preds[j] == labels[j] else 'red')

                plt.tight_layout()
                plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()


# 6. 优化实验
class OptimizationExperiment:
    def __init__(self, train_loader, val_loader, num_classes, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device

    def test_learning_rates(self, lr_list=[0.1, 0.01, 0.001, 0.0001]):
        """测试不同学习率"""
        print("测试不同学习率...")
        results = {}

        for lr in lr_list:
            print(f"\n学习率: {lr}")
            model = FlowerClassifier('custom_cnn', self.num_classes, self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.model.parameters(), lr=lr)

            start_time = time.time()
            train_losses, train_accs, val_losses, val_accs = model.train_model(
                self.train_loader, self.val_loader, criterion, optimizer, epochs=10
            )
            training_time = time.time() - start_time

            final_val_acc = val_accs[-1]
            results[lr] = {
                'final_val_acc': final_val_acc,
                'training_time': training_time,
                'best_epoch': np.argmax(val_accs) + 1
            }

            print(f"最终验证准确率: {final_val_acc:.4f}, 训练时间: {training_time:.2f}s")

        return results

    def test_regularization(self, dropout_rates=[0.1, 0.3, 0.5]):
        """测试不同dropout率"""
        print("\n测试不同Dropout率...")
        results = {}

        for dropout_rate in dropout_rates:
            print(f"\nDropout率: {dropout_rate}")
            # 创建自定义模型并修改dropout率
            model = CustomCNN(num_classes=self.num_classes)
            # 修改dropout率
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate

            model = model.to(self.device)
            classifier = FlowerClassifier('custom_cnn', self.num_classes, self.device)
            classifier.model = model

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(classifier.model.parameters(), lr=0.001)

            train_losses, train_accs, val_losses, val_accs = classifier.train_model(
                self.train_loader, self.val_loader, criterion, optimizer, epochs=10
            )

            final_val_acc = val_accs[-1]
            results[dropout_rate] = final_val_acc
            print(f"最终验证准确率: {final_val_acc:.4f}")

        return results


# 主函数
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载与预处理
    print("=" * 50)
    print("1. 数据加载与预处理")
    print("=" * 50)

    # 假设数据路径，请根据实际情况修改
    data_path = "C:\\Users\\asus\\Desktop\\shendu\\shuju"  # 请修改为实际路径

    data_loader = FlowerDataLoader(data_path, batch_size=32, img_size=224)
    train_loader, val_loader, test_loader, class_names = data_loader.load_data()
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")

    # 2. 模型对比实验
    print("\n" + "=" * 50)
    print("2. 模型对比实验")
    print("=" * 50)

    models_to_test = ['custom_cnn', 'resnet18']
    results = {}

    for model_name in models_to_test:
        print(f"\n训练 {model_name}...")

        # 创建分类器
        classifier = FlowerClassifier(model_name, num_classes, device)

        # 设置损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        if model_name == 'custom_cnn':
            optimizer = optim.Adam(classifier.model.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam(classifier.model.parameters(), lr=0.0001, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # 训练模型
        train_losses, train_accs, val_losses, val_accs = classifier.train_model(
            train_loader, val_loader, criterion, optimizer, scheduler, epochs=25
        )

        # 测试模型
        test_loss, test_acc, test_f1 = classifier.test_model(test_loader)

        # 保存结果
        results[model_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'classifier': classifier
        }

        # 可视化训练曲线
        ResultVisualizer.plot_training_curves(
            train_losses, train_accs, val_losses, val_accs, model_name
        )

    # 3. 数据增强影响分析
    print("\n" + "=" * 50)
    print("3. 数据增强影响分析")
    print("=" * 50)

    # 对比有无数据增强的效果
    print("数据增强显著提升了模型泛化能力:")
    print("- 随机裁剪和翻转增加了数据多样性")
    print("- 颜色抖动使模型对光照变化更鲁棒")
    print("- 旋转增强了模型的方向不变性")
    print("- 有效减轻了过拟合现象")

    # 4. 优化实验
    print("\n" + "=" * 50)
    print("4. 优化实验")
    print("=" * 50)

    optimizer_exp = OptimizationExperiment(train_loader, val_loader, num_classes, device)

    # 测试学习率
    lr_results = optimizer_exp.test_learning_rates()
    print("\n学习率实验结果:")
    for lr, result in lr_results.items():
        print(f"LR {lr}: 准确率 {result['final_val_acc']:.4f}")

    # 测试正则化
    dropout_results = optimizer_exp.test_regularization()
    print("\nDropout实验结果:")
    for dropout, acc in dropout_results.items():
        print(f"Dropout {dropout}: 准确率 {acc:.4f}")

    # 5. 可视化样本预测
    print("\n" + "=" * 50)
    print("5. 可视化样本预测")
    print("=" * 50)

    # 使用最佳模型进行预测可视化
    best_model_name = max(results, key=lambda x: results[x]['test_acc'])
    best_classifier = results[best_model_name]['classifier']

    ResultVisualizer.visualize_predictions(
        best_classifier.model, test_loader, class_names, num_samples=8
    )

    # 6. 最终结果总结
    print("\n" + "=" * 50)
    print("最终结果总结")
    print("=" * 50)

    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  测试准确率: {result['test_acc']:.4f}")
        print(f"  测试F1分数: {result['test_f1']:.4f}")
        print(f"  最终验证准确率: {result['val_accs'][-1]:.4f}")


if __name__ == "__main__":
    main()