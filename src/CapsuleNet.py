# *_*coding:utf-8 *_*

import torch
import pickle
from torch import nn
import numpy as np
import torch.utils.data as Data
from torchsummary import summary
import time
from torch.utils.checkpoint import checkpoint_sequential


NUM_CLASS = 4
LR = 0.001
eps = 1e-5
EPOCHS = 100
BATCH_SIZE = 64


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = torch.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)

    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules,
                 in_channels, out_channels,
                 kernel_size, strides):
        super(CapsuleLayer, self).__init__()

        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.capsules = nn.ModuleList(   # 73*3*8*8
            [nn.Conv2d(self.in_channels, self.out_channels,
                       kernel_size=self.kernel_size,
                       stride=self.strides, padding=0)
             for _ in range(self.num_capsules)]
        )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)

        return scale * tensor / torch.sqrt(squared_norm + eps)

    def forward(self, x):
        # [batch_size, 73*3*8, 1] * 8
        outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.squash(outputs)
        # print(outputs.shape)

        return outputs


class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, num_route_nodes,
                 in_channels, out_channels, num_iterations=3):
        super(DigitCapsules, self).__init__()

        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iterations = num_iterations

        self.route_weights = nn.Parameter(torch.randn(self.num_capsules,
                                                      self.num_route_nodes,
                                                      self.in_channels,
                                                      self.out_channels))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)

        return scale * tensor / torch.sqrt(squared_norm + eps)

    def forward(self, x):
        # None add 1 dimension
        # x = [batch_size, 73*3*8, 8]
        # self.rout_weights = [4, 73*3*8, 8, 16]
        # priors: return [4, batch_size, 73*3*8, 1, 16]
        priors = x[:, None, :, None, :] @ self.route_weights[None, :, :, :, :]
        # priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

        logits = torch.zeros(priors.size()).cuda()

        for i in range(self.num_iterations):
            prob = softmax(logits, dim=2)
            outputs = self.squash(torch.sum((prob * priors), dim=2, keepdim=True))

            if i != self.num_iterations - 1:
                logits += priors * outputs

        outputs = torch.squeeze(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=128,
                      kernel_size=(5, 3),
                      stride=1,
                      padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128, momentum=0.5),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(5, 3),
                      stride=1,
                      padding=(2, 1)),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(5, 3),
                      stride=1,
                      padding=(2, 1)),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(5, 3),
                      stride=1,
                      padding=(2, 1)),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(5, 3),
                      stride=1,
                      padding=(2, 1)),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(5, 3),
                      stride=1,
                      padding=(2, 1)),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
        )

        self.primaryCapsules = CapsuleLayer(
            num_capsules=8,
            in_channels=256,
            out_channels=8,
            kernel_size=5,
            strides=2
        )

        self.digitCapsules = DigitCapsules(
            num_capsules=NUM_CLASS,
            num_route_nodes=73*3*8,
            in_channels=8,
            out_channels=16
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.primaryCapsules(x)
        x = self.digitCapsules(x)

        classes = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1))

        return classes


def load_data(in_dir):
    with open(in_dir, 'rb') as f:
        train_data, train_label, \
        valid_data, valid_label, \
        test_data, test_label = pickle.load(f)

    return train_data, train_label, \
           valid_data, valid_label, \
           test_data, test_label


if __name__ == '__main__':
    start = time.time()
    capsNet = CapsuleNet()
    capsNet.cuda()
    summary(capsNet, (3, 300, 40))

    print('# parameters:', sum(param.numel() for param in capsNet.parameters()))

    optimizer = torch.optim.Adam(capsNet.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    train_data, train_label, \
    valid_data, valid_label, \
    test_data, test_label = load_data('../data/impro_IEMOCAP.pkl')


    train_data = torch.from_numpy(train_data.reshape((-1, 3, 300, 40)))
    train_label = torch.from_numpy(train_label).squeeze().long()
    valid_data = torch.from_numpy(valid_data.reshape((-1, 3, 300, 40)))
    valid_label = torch.from_numpy(valid_label).squeeze().long()
    test_data = torch.from_numpy(test_data.reshape((-1, 3, 300, 40)))
    test_label = torch.from_numpy(test_label).squeeze().long()

    train_size = train_data.size(0)
    valid_size = valid_data.size(0)
    test_size = test_data.size(0)


    torch_train_dataset = Data.TensorDataset(train_data, train_label)

    train_loader = Data.DataLoader(dataset=torch_train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    torch_valid_dataset = Data.TensorDataset(valid_data, valid_label)

    valid_loader = Data.DataLoader(dataset=torch_valid_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2)

    torch_test_dataset = Data.TensorDataset(test_data, test_label)

    test_loader = Data.DataLoader(dataset=torch_test_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2)


    for epoch in range(1, EPOCHS+1):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            pred = capsNet(batch_x)
            loss = loss_func(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_accuracy = torch.sum(torch.argmax(pred, dim=1) == batch_y).type(torch.FloatTensor) / batch_y.size(0)
            print('Epoch: {:0>3}'.format(epoch), '| step: {:0>2}/{}'.format(step, len(train_loader)),
                  '| loss: {:.2f}'.format(loss.detach().cpu().numpy()),
                  '| train accuracy: {:.2%}'.format(train_accuracy.detach().cpu().numpy()))


            if step == len(train_loader) - 1:

                capsNet.eval()

                with torch.no_grad():

                    train_loss = 0.
                    train_accuracy = 0.

                    for (batch_train_x, batch_train_y) in train_loader:
                        batch_train_x = batch_train_x.cuda()
                        batch_train_y = batch_train_y.cuda()

                        train_pred = capsNet(batch_train_x)
                        b_train_loss = loss_func(train_pred, batch_train_y)
                        train_loss += b_train_loss

                        train_pred = torch.argmax(train_pred, dim=1)

                        b_train_accuracy = torch.sum(batch_train_y == train_pred).type(
                            torch.FloatTensor) / batch_train_y.size(0)
                        train_accuracy += b_train_accuracy

                    train_loss /= len(train_loader)
                    train_accuracy /= len(train_loader)

                    print('------------Epoch: {:0>3}'.format(epoch),
                          '| train loss: {:.2f}'.format(train_loss),
                          '| train accuracy: {:.2%}'.format(train_accuracy))

                capsNet.train()

            if epoch % 10 == 0 and step == len(train_loader)-1:
                torch.save(capsNet, '../model/model_{:0>2}.pkl'.format(epoch))

                capsNet.eval()

                with torch.no_grad():

                    valid_loss = 0.
                    valid_accuracy = 0.

                    for (batch_valid_x, batch_valid_y) in valid_loader:
                        batch_valid_x = batch_valid_x.cuda()
                        batch_valid_y = batch_valid_y.cuda()


                        valid_pred = capsNet(batch_valid_x)
                        b_valid_loss = loss_func(valid_pred, batch_valid_y)
                        valid_loss += b_valid_loss

                        valid_pred = torch.argmax(valid_pred, dim=1)

                        b_valid_accuracy = torch.sum(batch_valid_y == valid_pred).type(torch.FloatTensor) / batch_valid_y.size(0)
                        valid_accuracy += b_valid_accuracy

                    valid_loss /= len(valid_loader)
                    valid_accuracy /= len(valid_loader)

                    print('------------Epoch: {:0>3}'.format(epoch),
                          '| cross_validation loss: {:.2f}'.format(valid_loss),
                          '| cross_validation accuracy: {:.2%}'.format(valid_accuracy))

                capsNet.train()

    test_loss = 0.
    test_accuracy = 0.
    capsNet.eval()
    with torch.no_grad():
        for batch_test_x, batch_test_y in test_loader:
            batch_test_x = batch_test_x.cuda()
            batch_test_y = batch_test_y.cuda()

            test_pred = capsNet(batch_test_x)
            b_test_loss = loss_func(test_pred, batch_test_y)
            test_loss += b_test_loss

            test_pred = torch.argmax(test_pred, dim=1)

            b_test_accuracy = torch.sum(batch_test_y == test_pred).type(torch.FloatTensor) / batch_test_y.size(0)
            test_accuracy += b_test_accuracy

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        print('------------test loss: {:.2f}'.format(test_loss),
              '| test accuracy: {:.2%}'.format(test_accuracy))

    end = time.time()
    print('总用时：{:.2f}mins'.format((end - start) / 60))
