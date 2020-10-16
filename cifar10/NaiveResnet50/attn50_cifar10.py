
#Library Utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models

import math
import os
from tqdm import tqdm
import shutil


# In[37]:


#Attention Layers
#Type 1
#Convolution Attention
class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        #Correction 1
        #v_out_h, v_out_w = v_out.split(self.out_channels // 2, dim=1)
        #v_out = torch.cat((v_out_h + self.rel_h, v_out_w + self.rel_w), dim=1)
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        #Correction 2
        out = q_out * k_out
        #out = (q_out * k_out).sum(dim=2)
        
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

# temp = torch.randn((2, 3, 32, 32))
# conv = AttentionConv(3, 16, kernel_size=3, padding=1)
# print(conv(temp).size())


# In[38]:


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


# In[39]:


class Model(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, stem=False):
        super(Model, self).__init__()
        self.in_places = 64

        if stem:
            self.init = nn.Sequential(
                # CIFAR10
                AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # For ImageNet
                # AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(4, 4)
            )
        else:
            self.init = nn.Sequential(
                # CIFAR10
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # For ImageNet
                # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dense = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride))
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def ATNnet26(num_classes=10, stem=False):
    return Model(Bottleneck, [1, 2, 4, 1], num_classes=num_classes, stem=stem)


def ATNnet38(num_classes=10, stem=False):
    return Model(Bottleneck, [2, 3, 5, 2], num_classes=num_classes, stem=stem)


def ATNnet50(num_classes=10, stem=False):
    return Model(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem)


# In[40]:


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


# In[41]:


EPOCHS=100
INIT_LR=1e-1

num_workers=8
img_size=32
batch_size=25
momentum=0.9
weight_decay=5e-4
num_classes = 10
model_name='Resnet50'

print_interval=500

# In[42]:


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = INIT_LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()

    train_acc = 0.0
    step = 0
    for data, target in train_loader:
        adjust_learning_rate(optimizer,  epoch)
        
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % print_interval == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            #logger.info("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epochs, loss.data, acc))
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))
                #logger.info("Current learning rate is: {}".format(param_group['lr']))


def eval(model, test_loader):
    print('evaluation ...')
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}'.format(acc))
    return acc


# In[ ]:





# In[43]:


# Data loading code
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, download=True, transform=transform_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)


# In[44]:


def save_checkpoint(state, is_best, filename):
    file_path = os.path.join('./checkpoint', filename)
    torch.save(state, file_path)
    best_file_path = os.path.join('./checkpoint', 'best_' + filename)
    if is_best:
        print('best Model Saving ...')
        shutil.copyfile(file_path, best_file_path)


# In[45]:


print('img_size: {}, num_classes: {}, stem: {}'.format(img_size, num_classes, True))

if model_name=='ATNnet50': 
    print("Running ATNnet50")
    model = ATNnet50(num_classes=num_classes, stem=True)
else:
    print("Running Resnet50")
    model = models.resnet50()

print("Number of model parameters: ", get_model_parameters(model))

start_epoch = 1
best_acc = 0.0

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=momentum, weight_decay=weight_decay)

for epoch in range(start_epoch, EPOCHS + 1):
    train(model, train_loader, optimizer, criterion, epoch)
    eval_acc = eval(model, test_loader)

    is_best = eval_acc > best_acc
    best_acc = max(eval_acc, best_acc)

    if is_best:
        print("printing best accuracy so far{}".format(best_acc))
        
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    filename = 'model_ckpt.tar'
    print('filename :: ', filename)

    parameters = get_model_parameters(model)

    if torch.cuda.device_count() > 1:
        save_checkpoint({
            'epoch': epoch,
            'arch': model_name,
            'state_dict': model.module.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'parameters': parameters,
        }, is_best, filename)
    else:
        save_checkpoint({
            'epoch': epoch,
            'arch': model_name,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'parameters': parameters,
        }, is_best, filename)
