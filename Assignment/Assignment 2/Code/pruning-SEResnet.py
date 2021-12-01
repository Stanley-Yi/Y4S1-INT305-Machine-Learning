from __future__ import print_function
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
from torchsummary import summary
from torchvision import datasets, transforms
import argparse
import torch_pruning as tp


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Net():
    model = SENet(PreActBlock, [3, 4, 6, 3])
    return model


def train(args, model,device,train_loader,optimizer,epoch,pruning_modules):
    model.train().to(device)
    correct = 0
    criteration = nn.CrossEntropyLoss()
    for i,(x,y) in enumerate(train_loader):
        x , y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss =  criteration(output,y)
        loss.backward()
        optimizer.step()

        if True:
            updateBN(model,0.0001,pruning_modules)



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
    plt.title('Confusion matrix: pruning-SEResnet')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path + "/Confusion matrix")
    plt.show()


def updateBN(model, s ,pruning_modules):
    for module in pruning_modules:
        module.weight.grad.data.add_(s * torch.sign(module.weight.data))

def get_pruning_modules(model):
    module_list = []
    for module in model.modules():
        if isinstance(module,PreActBlock):
            module_list.append(module.bn1)
            module_list.append(module.bn2)
    return module_list


def gather_bn_weights(model,pruning_modules):
    size_list = [module.weight.data.shape[0] for module in model.modules() if module in pruning_modules]
    # print(size_list)
    bn_weights = torch.zeros(sum(size_list))
    # print(bn_weights)
    index = 0
    for module, size in zip(pruning_modules, size_list):
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index += size

    return bn_weights

def computer_eachlayer_pruned_number(bn_weights,thresh, bn_modules):
    num_list = []
    #print(bn_modules)
    for module in bn_modules:
        num = 0
        #print(module.weight.data.abs(),thresh)
        for data in module.weight.data.abs():
            if thresh > data.float():
                num +=1
        num_list.append(num)
    print(thresh)
    return num_list

def prune_model(model, num_list):
    model.to('cuda')
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 1, 28, 28))

    def prune_bn(bn, num):
        L1_norm = bn.weight.detach().cpu().numpy()
        prune_index = np.argsort(L1_norm)[:num].tolist()  # remove filters with small L1-Norm
        plan = DG.get_pruning_plan(bn, tp.prune_batchnorm, prune_index)
        print(plan)
        plan.exec()

    blk_id = 0
    for m in model.modules():
        if isinstance(m, PreActBlock):
            prune_bn(m.bn1, num_list[blk_id])
            prune_bn(m.bn2, num_list[blk_id + 1])
            blk_id += 2
    return model



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
    transform = transforms.Compose([
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
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Put the model on the GPU or CPU
    model = Net().to(device)
    model.load_state_dict(torch.load('./model_result/senet+light resnet/16_99.58.pt'))


    # Begin training and testing
    now = str(datetime.datetime.now()).split('.')[0].replace(":", "-")
    base_path = "./model_result/pruning-SEResnet/"
    os.makedirs(base_path + now)
    best_accuracy = [-1, 0.]
    loss_list = []
    accuracy_list = []

    summary(model.to(device), (1, 28, 28))

    # https://blog.csdn.net/qq_38109843/article/details/107671873?spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-11.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-11.no_search_link
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)

    bn_modules = get_pruning_modules(model)

    bn_weights = gather_bn_weights(model, bn_modules)
    print(bn_weights)
    sorted_bn, sorted_index = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * 0.04)
    thresh = sorted_bn[thresh_index].to(device)
    # print("fsaf", thresh)

    num_list = computer_eachlayer_pruned_number(bn_weights, thresh, bn_modules)

    prune_model(model, num_list)
    # print('sfsdfsfs', model)

    res = test(model.to(device), device, test_loader)
    torch.save(model, base_path + str(now) + "/" + str(res[1]) + ".pt")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, bn_modules)
        res = test(model.to(device), device, test_loader)
        loss_list.append(res[0])
        accuracy_list.append(round(res[1] / 100, 2))
        torch.save(model, base_path + str(now) + "/" + str(res[1]) + ".pt")


    print(best_accuracy)
    summary(model.to(device), (1, 28, 28))

    plt.plot(range(1, args.epochs + 1, 1), loss_list, color='#66B3FF', linestyle='-', label='loss')
    plt.plot(range(1, args.epochs + 1, 1), accuracy_list, color='#FF9224', linestyle='-', label='acc')
    # plt.text(best_accuracy[0], best_accuracy[1], str(best_accuracy[1]))
    plt.title('loss/acc graph: pruning-SEResnet')
    plt.ylabel("Test: loss/acc")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.legend()
    plt.savefig(base_path + str(now) + "/loss_acc")
    plt.show()

    # confusion matrix
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
            pic_tensor = torch.masked_select(batch_images,
                                             mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 28, 28))
            info_tensor = torch.masked_select(out, mask.unsqueeze(1).expand(-1, out.shape[1]))
            prediction_tensor = torch.masked_select(prediction, mask)
            labels_tensor = torch.masked_select(batch_labels, mask)
            # print(info_list.shape, labels_list.shape, pic_tensor.shape)
            for index in range(len(labels_tensor)):
                if int(labels_tensor[index]) in mis_classify:
                    continue
                else:
                    mis_info = [prediction_tensor[index]]
                    confidence = max(info_tensor[index * 10:index * 10 + 10])
                    mis_info.append(confidence)
                    # print(confidence)
                    pic = pic_tensor[index * 28 * 28:(index + 1) * 28 * 28].reshape(1, 1, 28, 28).squeeze()
                    mis_info.append(pic)
                    mis_classify[int(labels_tensor[index])] = mis_info

            key_list = list(mis_classify.keys())
            key_list.sort()
            if key_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                mis_flag = False

        if well_flag:
            mask = prediction.eq(batch_labels)
            # print(mask.shape)
            pic_tensor = torch.masked_select(batch_images,
                                             mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 28, 28))
            info_tensor = torch.masked_select(out, mask.unsqueeze(1).expand(-1, out.shape[1]))
            prediction_tensor = torch.masked_select(prediction, mask)
            labels_tensor = torch.masked_select(batch_labels, mask)
            # print(info_list.shape, labels_list.shape, pic_tensor.shape)
            for index in range(len(labels_tensor)):
                if int(labels_tensor[index]) in well_classify:
                    continue
                else:
                    well_info = [prediction_tensor[index]]
                    confidence = max(info_tensor[index * 10:index * 10 + 10])
                    well_info.append(confidence)
                    # print(confidence)
                    pic = pic_tensor[index * 28 * 28:(index + 1) * 28 * 28].reshape(1, 1, 28, 28).squeeze()
                    well_info.append(pic)
                    well_classify[int(labels_tensor[index])] = well_info

            key_list = list(well_classify.keys())
            key_list.sort()
            if key_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                well_flag = False

    plot_confusion_matrix(conf_matrix.numpy(), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          save_path=base_path + str(now), normalize=False, title='Normalized confusion matrix')

    # show mis classification
    for key, value in mis_classify.items():
        plt.subplot(2, 5, key + 1)
        plt.title('prediction: ' + str(int(value[0])))
        plt.imshow(value[2].cpu(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('confidence: ' + str(round(float(value[1]), 2)))
    plt.tight_layout()
    plt.suptitle('Mis-classified example: pruning-SEResnet')
    plt.savefig(base_path + str(now) + "/mis classification")
    plt.show()

    for key, value in well_classify.items():
        plt.subplot(2, 5, key + 1)
        plt.title('prediction: ' + str(int(value[0])))
        plt.imshow(value[2].cpu(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('confidence: ' + str(round(float(value[1]), 2)))
    plt.tight_layout()
    plt.suptitle('Well-classified example: pruning-SEResnet')
    plt.savefig(base_path + str(now) + "/well classification")
    plt.show()


if __name__ == '__main__':
    main()