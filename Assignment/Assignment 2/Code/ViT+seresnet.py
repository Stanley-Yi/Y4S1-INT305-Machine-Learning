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


# source code https://www.jianshu.com/p/06a40338dc7c

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 4, emb_size: int = 768, img_size: int = 28):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        # print("1qkv's shape: ", self.qkv(x).shape)
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # print("2qkv's shape: ", qkv.shape)

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print("queries's shape: ", queries.shape)
        # print("keys's shape: ", keys.shape)
        # print("values's shape: ", values.shape)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        # print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        # print("scaling: ", scaling)
        att = F.softmax(energy, dim=-1) / scaling
        # print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        # print("att2' shape: ", att.shape)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        # print("out2's shape: ", out.shape)
        out = self.projection(out)
        # print("out3's shape: ", out.shape)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class Net(nn.Sequential):
    def __init__(self,
                in_channels: int = 1,
                patch_size: int = 4,
                emb_size: int = 768,
                img_size: int = 28,
                depth: int = 12,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


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
    plt.title('Confusion matrix: ViT+se_resnet')
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
    base_path = "./model_result/ViT+se_resnet/"
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
    summary(Net().to(device), (1, 28, 28))

    plt.plot(range(1, args.epochs + 1, 1), loss_list, color='#66B3FF', linestyle='-', label='loss')
    plt.plot(range(1, args.epochs + 1, 1), accuracy_list, color='#FF9224', linestyle='-', label='acc')
    plt.title('loss/acc graph: ViT+se_resnet')
    plt.ylabel("Test: loss/acc")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.legend()
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
    plt.suptitle('Mis-classified example: ViT+se_resnet')
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
    plt.suptitle('Well-classified example: ViT+se_resnet')
    plt.savefig(base_path + str(now) + "/well classification")
    plt.show()


if __name__ == '__main__':
    main()