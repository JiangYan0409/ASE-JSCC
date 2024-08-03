import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18  # 以ResNet18为例，也可以根据实际情况选择其他模型
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 生成高斯噪音
mean = 0  # 均值
std_dev = 0.1  # 标准差，根据需要调整
train_dataset = datasets.ImageFolder(root='data/UCMerced_LandUse-train/Images', transform=transform)
test_dataset = datasets.ImageFolder(root='data/UCMerced_LandUse-test/Images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模拟通信线路
# The (real) AWGN channel    
def AWGN_channel(x, snr, P=2):
    batch_size, channels, height, width = x.shape  
    gamma = 10 ** (snr / 10.0)      
    noise = torch.sqrt(P / gamma) * torch.randn(batch_size, channels, height, width).to(device)  
    y = x + noise   
    return y


# Please set the symbol power if it is not a default value
def Fading_channel(x, snr, P = 2):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    h_I = torch.randn(batch_size, K).to(device)
    h_R = torch.randn(batch_size, K).to(device) 
    h_com = torch.complex(h_I, h_R)  
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_com = h_com*x_com
    
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).to(device)
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).to(device)
    noise = torch.complex(n_I, n_R)
    
    y_add = y_com + noise
    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).to(device)
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out

# 联合Fading和AWGN信道
def Combined_channel(x, snr, batch_size, channel, height, width):
    P=2
    # 首先通过Fading信道
    x_faded = Fading_channel(x, snr, P)
    print ("x_faded.shape:",x_faded.shape)
    # 然后通过AWGN信道
    x_faded = x_faded.view((batch_size, channel, height, width))
    print ("x_faded.view.shape:",x_faded.shape)
    snr = torch.randint(0, 28, (x_faded.shape[0], x_faded.shape[1], x_faded.shape[2], 1)).to(device)
    x_combined = AWGN_channel(x_faded, snr, P)
    return x_combined


def Channel(z, snr, channel_type, batch_size, channel, height, width):
    if channel_type == 'AWGN':
        z = AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        z = Fading_channel(z, snr)
    elif channel_type == 'Combined_channel':
        z = Combined_channel(z, snr, batch_size, channel, height, width)
    return z

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 调整 stride 为 1，不改变大小
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten() 
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, channel_type):
        # x = self.encoder(x)
        # print("encoder x.shape:",x.shape)
        # noise = torch.randn_like(x) * std_dev + mean
        # x = x + noise
        batch_size, channel, height, width = x.shape
        if channel_type == 'Fading' or channel_type == 'Combined_channel':
            x = self.flatten(x)
            print("after flatten x.shape", x.shape)
            SNR = torch.randint(0, 28, (x.shape[0], 1)).to(device)
        else :
            SNR = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1)).to(device)
        x = Channel(x, SNR, channel_type, batch_size, channel, height, width)
        print("after Channel x.shape:",x.shape)
        x = x.view((batch_size, channel, height, width))     
        # x = self.decoder(x)
        return x



def mask_gen(weights, cr):
    position = round(cr*weights.size(1))
    weights_sorted, index = torch.sort(weights, dim=1)
    mask = torch.zeros_like(weights)

    for i in range(weights.size(0)):
        weight = weights_sorted[i, position-1]
        # print(weight)
        for j in range(weights.size(1)):
            if weights[i, j] <= weight:
                mask[i, j] = 1
    return mask  
  

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x, cr=0.8):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        print("y shape of Fsq:", y.shape)
        y = self.fc(y)
        print("y shape of Fex:", y.shape)
        mask = mask_gen(y, cr).view(b,c,1,1)
        print("mask shape:", mask.shape)
        print("x shape:", x.shape)
        return x * mask


class SatelliteClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(SatelliteClassifierWithAttention, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.attention_module = SE_Block(in_features)
        self.antoencoder = Autoencoder()
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, cr, channel_type):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        print("before x.shape:",x.shape)
        # print("before x:",x)
        x = self.attention_module(x, cr)
        print("after attention_module x.shape:",x.shape)

        x = self.antoencoder(x, channel_type)
        print("after antoencoder x.shape:",x.shape)
        x = self.resnet18.avgpool(x)
        # in_features_fc = x.size(1)
        # self.resnet18.fc = nn.Linear(in_features_fc, num_classes)
        x = x.view(x.size(0), -1)
        x = self.resnet18.fc(x)

        return x

def continue_train(cr, num_epochs, pre_checkpoint, channel_type):
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    num_classes = len(train_dataset.classes)  
    model = SatelliteClassifierWithAttention(num_classes)
    model = model.to(device)

    pretrained_dict = torch.load(f'{pre_checkpoint}')
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    # num_epochs = 50 # 或者你想要的训练轮数

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, cr, channel_type)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        writer.add_scalar('Training Loss', running_loss/len(train_loader), epoch + 1)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, cr, channel_type)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')

        writer.add_scalar('Test Accuracy', accuracy)
# Save the model with the specified cr and num_epochs in the file name
    save_path = f'checkpoint/classifier_attention_auto_UCMerced_LandUse_{channel_type}_ResNet18_60epoch_0.5_up_{num_epochs}epoch_{cr}.pth'
    torch.save(model.state_dict(), save_path)

    writer.close()

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # Write results to a txt file
    with open(f'logs/ResNet18/classifier_attention_auto_UCMerced_LandUse_{channel_type}_ResNet18_60_up_{num_epochs}epoch_{cr}.txt', 'w') as file:
        file.write('strat comtinue training...\n')
        file.write(f'Time: {start_time}----------{current_time}\n')
        file.write(f'model name:{save_path}\n')
        file.write(f'channel_type:{channel_type}\n')
        file.write(f'CR (Compression Ratio): {cr}\n')
        file.write(f'Num Epochs: {num_epochs}\n')
        file.write(f'Test Accuracy: {accuracy}\n')
        file.write('train over!\n')

def train(cr, num_epochs, channel_type):
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    num_classes = len(train_dataset.classes)  # 类别数
    model = SatelliteClassifierWithAttention(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    writer = SummaryWriter()
    # num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            print("epoch:",epoch)
            print(images.shape)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, cr, channel_type)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
 
        avg_train_loss = running_loss / len(train_loader)

        scheduler.step(avg_train_loss)
        writer.add_scalar('Training Loss', running_loss/len(train_loader), epoch + 1)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, cr, channel_type)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')
        writer.add_scalar('Test Accuracy', accuracy)

    # Save the model with the specified cr and num_epochs in the file name
    save_path = f'checkpoint/classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_{num_epochs}epoch_{cr}.pth'
    torch.save(model.state_dict(), save_path)

    
    writer.close()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # Write results to a txt file
    with open(f'logs/ResNet18/classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_{num_epochs}epoch_{cr}.txt', 'w') as file:
        file.write('strat training...\n')
        file.write(f'Time: {start_time}----------{current_time}\n')
        file.write(f'model name:{save_path}\n')
        file.write(f'model name:{channel_type}\n')
        file.write(f'CR (Compression Ratio): {cr}\n')
        file.write(f'Num Epochs: {num_epochs}\n')
        file.write(f'Test Accuracy: {accuracy}\n')
        file.write('train over!\n')

def main(task,cr,num_epochs,pre_checkpoint,channel_type):
    if task == 'continue':
        print("continue_train start!")
        continue_train(cr, num_epochs,pre_checkpoint,channel_type)
        print("continue_train over!")
    else :
        print("train start!")
        train(cr, num_epochs,channel_type)
        print("train over!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or continue training a model.")
    parser.add_argument('--task', choices=['continue', 'train'], default='train', required=True, help='Specify the task (continue or train).')
    parser.add_argument('--cr', type=float, default=0.1, help='Specify the compression ratio (cr) for the SE Block.')
    parser.add_argument('--num_epochs', type=int, default=60, help='Specify the number of epochs for training.')
    parser.add_argument('--pre_checkpoint', type=str, default=None, help='Specify the pretrained checkpoint for continue train.')
    parser.add_argument('--channel_type', choices=['AWGN', 'Fading',"Combined_channel"], default='Combined_channel', help='Specify the channel_type for transfer.')
    args = parser.parse_args()
    main(args.task, args.cr, args.num_epochs,args.pre_checkpoint,args.channel_type)

