from __future__ import print_function

from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime
import itertools
import os
import numpy as np
from torchsummary import summary
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import warnings
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any
from collections import namedtuple



def googlenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> "GoogLeNet":

    return GoogLeNet(**kwargs)


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 10,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        conv_block = blocks[0]
        inception_block = blocks[1]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 1 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    def forward(self, x: Tensor):
        x = self._transform_input(x)
        x = self._forward(x)
        return x


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


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
    plt.title('Confusion matrix: GoogLeNet')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path + "/Confusion matrix")
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
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
    model = googlenet().to(device)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Begin training and testing
    now = str(datetime.datetime.now()).split('.')[0].replace(":", "-")
    base_path = "./model_result/GoogLeNet/"
    os.makedirs(base_path+now)
    best_accuracy = [-1, 0.]
    loss_list = []
    accuracy_list = []
    for epoch in range(1, args.epochs + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
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
    summary(googlenet().to(device), (1, 28, 28))

    plt.plot(range(1, args.epochs + 1, 1), loss_list, color='#66B3FF', linestyle='-', label='loss')
    plt.plot(range(1, args.epochs + 1, 1), accuracy_list, color='#FF9224', linestyle='-', label='acc')
    plt.title('loss/acc graph: GoogLeNet')
    plt.ylabel("Test: loss/acc")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(base_path + str(now) + "/loss_acc")
    plt.show()

    # model = Net().to(device)
    # model.load_state_dict(torch.load('./model_result/attention+reduce resnet/2021-11-22 19-59-55/10_94.63.pt'))

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
    plt.suptitle('Mis-classified example: GoogLeNet')
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
    plt.suptitle('Well-classified example: GoogLeNet')
    plt.savefig(base_path + str(now) + "/well classification")
    plt.show()


if __name__ == '__main__':
    main()