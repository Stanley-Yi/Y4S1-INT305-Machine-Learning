from __future__ import print_function
import argparse
import datetime
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Adjust the model to get a higher performance
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # shortcut connection
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def Net():
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    return model


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    plt.figure()
    pic = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx in (1,2,3,4,5):
            if batch_idx == 1:
                pic = data[0,0,:,:]
            else:
                pic = torch.cat((pic,data[0,0,:,:]),dim=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # Calculate gradients
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    plt.imshow(pic.cpu(), cmap='gray')
    plt.show()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.title('Confusion matrix: reduce size+resnet')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path + "/Confusion matrix")
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalize the input (black and white image)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Make train dataset split
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    # Make test dataset split
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    # Convert the dataset to dataloader, including train_kwargs and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Put the model on the GPU or CPU
    model = Net().to(device)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)

    # Begin training and testing
    now = str(datetime.datetime.now()).split('.')[0].replace(":", "-")
    base_path = "./model_result/reduce size+resnet/"
    os.makedirs(base_path+now)
    best_accuracy = [-1, 0.]
    loss_list = []
    accuracy_list = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        res = test(model, device, test_loader)
        loss_list.append(res[0])
        accuracy_list.append(round(res[1]/100, 2))
        torch.save(model.state_dict(), base_path+str(now)+"/"+str(epoch)+"_"+str(res[1])+".pt")
        if res[1] > best_accuracy[1]:
            best_accuracy[1] = res[1]
            best_accuracy[0] = epoch
        scheduler.step()

    print(best_accuracy)
    summary(Net().to(device), (1, 28, 28))

    plt.plot(range(1, args.epochs + 1, 1), loss_list, color='#66B3FF', linestyle='-', label='loss')
    plt.plot(range(1, args.epochs + 1, 1), accuracy_list, color='#FF9224', linestyle='-', label='acc')
    # plt.text(best_accuracy[0], best_accuracy[1], str(best_accuracy[1]))
    plt.title('loss/acc graph: reduce size+resnet')
    plt.ylabel("Test: loss/acc")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.legend()
    plt.savefig(base_path + str(now) + "/loss_acc")
    plt.show()

    # confusion matrix
    # source code see https://blog.csdn.net/qq_18617009/article/details/103345308
    mis_classify = {}
    well_classify = {}
    mis_flag = True
    well_flag = True
    conf_matrix = torch.zeros(10, 10)
    for batch_images, batch_labels in test_loader:
        # print(batch_labels)
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

        out = F.softmax(model(batch_images), 1)
        # print(out.shape)  # torch.Size([1000, 10])
        prediction = torch.max(out, 1)[1]
        # print(prediction.shape)  # torch.Size([1000])
        conf_matrix = confusion_matrix(prediction, labels=batch_labels, conf_matrix=conf_matrix)

        if mis_flag:
            mask = prediction.eq(batch_labels).logical_not()
            # print(mask.shape)
            pic_tensor = torch.masked_select(batch_images, mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 28, 28))
            info_tensor = torch.masked_select(out, mask.unsqueeze(1).expand(-1, out.shape[1]))
            prediction_tensor = torch.masked_select(prediction, mask)
            labels_tensor = torch.masked_select(batch_labels, mask)
            # print(info_list.shape, labels_list.shape, pic_tensor.shape)
            for index in range(len(labels_tensor)):
                if int(labels_tensor[index]) in mis_classify:
                    continue
                else:
                    mis_info = [prediction_tensor[index]]
                    confidence = max(info_tensor[index*10:index*10+10])
                    mis_info.append(confidence)
                    # print(confidence)
                    pic = pic_tensor[index*28*28:(index+1)*28*28].reshape(1, 1, 28, 28).squeeze()
                    mis_info.append(pic)
                    mis_classify[int(labels_tensor[index])] = mis_info

            key_list = list(mis_classify.keys())
            key_list.sort()
            if key_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                mis_flag = False

        if well_flag:
            mask = prediction.eq(batch_labels)
            # print(mask.shape)
            pic_tensor = torch.masked_select(batch_images, mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 28, 28))
            info_tensor = torch.masked_select(out, mask.unsqueeze(1).expand(-1, out.shape[1]))
            prediction_tensor = torch.masked_select(prediction, mask)
            labels_tensor = torch.masked_select(batch_labels, mask)
            # print(info_list.shape, labels_list.shape, pic_tensor.shape)
            for index in range(len(labels_tensor)):
                if int(labels_tensor[index]) in well_classify:
                    continue
                else:
                    well_info = [prediction_tensor[index]]
                    confidence = max(info_tensor[index*10:index*10+10])
                    well_info.append(confidence)
                    # print(confidence)
                    pic = pic_tensor[index*28*28:(index+1)*28*28].reshape(1, 1, 28, 28).squeeze()
                    well_info.append(pic)
                    well_classify[int(labels_tensor[index])] = well_info

            key_list = list(well_classify.keys())
            key_list.sort()
            if key_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                well_flag = False

    plot_confusion_matrix(conf_matrix.numpy(), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], save_path=base_path + str(now), normalize=False, title='Normalized confusion matrix')

    # show mis classification
    for key, value in mis_classify.items():
        plt.subplot(2, 5, key+1)
        plt.title('prediction: ' + str(int(value[0])))
        plt.imshow(value[2].cpu(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('confidence: ' + str(round(float(value[1]), 2)))
    plt.tight_layout()
    plt.suptitle('Mis-classified example: reduce size+resnet')
    plt.savefig(base_path + str(now) + "/mis classification")
    plt.show()

    for key, value in well_classify.items():
        plt.subplot(2, 5, key+1)
        plt.title('prediction: ' + str(int(value[0])))
        plt.imshow(value[2].cpu(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('confidence: ' + str(round(float(value[1]), 2)))
    plt.tight_layout()
    plt.suptitle('Well-classified example: reduce size+resnet')
    plt.savefig(base_path + str(now) + "/well classification")
    plt.show()


if __name__ == '__main__':
    main()