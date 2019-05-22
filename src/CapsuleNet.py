# *_*coding:utf-8 *_*

import torch
import pickle
from torch import nn
import numpy as np
import torch.utils.data as Data
from torchsummary import summary
import time
from sklearn.metrics import confusion_matrix as confusion


NUM_CLASS = 4
LR = 0.001
eps = 1e-7
EPOCHS = 200
BATCH_SIZE = 64


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = torch.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)

    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.strides,
                      padding=self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.strides,
                      padding=self.padding),
            nn.BatchNorm2d(self.out_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)

        return out


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size,
                            self.num_layers, batch_first=True,
                            dropout = 0.5, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out


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

        self.capsules = nn.ModuleList(   # 73*8*8*8
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
        # [batch_size, 73*8*8, 1] * 8
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
        # x = [batch_size, 73*8*8, 8]
        # self.rout_weights = [4, 73*8*8, 8, 16]
        # priors: return [4, batch_size, 73*8*8, 1, 16]
        priors = x[:, None, :, None, :] @ self.route_weights[None, :, :, :, :]

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
                      out_channels=64,
                      kernel_size=(7, 1),
                      stride=1,
                      padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 1),
                      stride=1,
                      padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(1, 5),
                      stride=1,
                      padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.redidualBlock1 = ResidualBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=(1, 5),
            strides=1,
            padding=(0, 2)
        )

        self.redidualBlock2 = ResidualBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=(1, 3),
            strides=1,
            padding=(0, 1)
        )

        self.blstm = BLSTM(
            input_size=20*128,
            hidden_size=128,
            num_layers=3,
            num_classes=4
        )

        self.primaryCapsules = CapsuleLayer(
            num_capsules=8,
            in_channels=128,
            out_channels=8,
            kernel_size=5,
            strides=2
        )

        self.digitCapsules = DigitCapsules(
            num_capsules=NUM_CLASS,
            num_route_nodes=73*8*8,
            in_channels=8,
            out_channels=16
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        to_lstm = x.permute(0, 2, 1, 3)
        to_lstm = to_lstm.contiguous().view(-1, to_lstm.size(1),
                                            to_lstm.size(2)*to_lstm.size(3))
        to_lstm = self.blstm(to_lstm)

        # to_caps = self.primaryCapsules(x)
        # to_caps = self.digitCapsules(to_caps)
        #
        # to_caps = torch.sqrt(torch.sum(torch.pow(to_caps, 2), dim=-1))
        #
        # output = to_lstm + to_caps
        #
        # return output
        return to_lstm


def load_data(in_dir):
    with open(in_dir, 'rb') as f:
        train_data, train_label, \
        valid_data, valid_label, \
        test_data, test_label = pickle.load(f)

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label).squeeze().long()
    valid_data = torch.from_numpy(valid_data)
    valid_label = torch.from_numpy(valid_label).squeeze().long()
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label).squeeze().long()

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

    return train_loader, valid_loader, test_loader


def train(capsNet, optimizer, loss_func, train_loader, valid_loader):

    for epoch in range(1, EPOCHS + 1):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            pred = capsNet(batch_x)
            loss = loss_func(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(pred, dim=1)

            train_accuracy = torch.sum(pred == batch_y).type(torch.FloatTensor) / batch_y.size(0)
            print('Epoch: {:0>3}/{}'.format(epoch, EPOCHS),
                  '| step: {:0>2}/{}'.format(step+1, len(train_loader)),
                  '| loss: {:.2f}'.format(loss.detach().cpu().numpy()),
                  '| train accuracy: {:.2%}'.format(train_accuracy.detach().cpu().numpy()))

            if step == len(train_loader) - 1:

                capsNet.eval()

                with torch.no_grad():

                    train_loss = 0.
                    train_result = []
                    train_label = []


                    for (batch_train_x, batch_train_y) in train_loader:
                        batch_train_x = batch_train_x.cuda()
                        batch_train_y = batch_train_y.cuda()

                        train_pred = capsNet(batch_train_x)
                        b_train_loss = loss_func(train_pred, batch_train_y)
                        train_loss += float(b_train_loss)

                        train_pred = torch.argmax(train_pred, dim=1)
                        train_result.extend(train_pred.detach().cpu().numpy())
                        train_label.extend(batch_train_y.detach().cpu().numpy())

                    train_weighted_accuracy = weighted_accuracy(train_result, train_label)
                    train_unweighted_accuracy = unweighted_accuracy(train_result, train_label)
                    train_confusion = confusion(train_label, train_result,
                                                labels=list(range(4)))

                    train_loss /= len(train_loader)

                    print('------------Epoch: {:0>3}/{}'.format(epoch, EPOCHS),
                          '| train loss: {:.2f}'.format(train_loss),
                          '| train weighted accuracy: {:.2%}'.format(train_weighted_accuracy),
                          '| train unweighted accuracy: {:.2%}'.format(train_unweighted_accuracy),
                          '\ntrain confusion matrix:\n', train_confusion)

                capsNet.train()

            if epoch % 10 == 0 and step == len(train_loader) - 1:
                torch.save(capsNet, '../model/model_{:0>2}.pkl'.format(epoch))
                print('save the intermediate model successfully')

                capsNet.eval()

                with torch.no_grad():

                    valid_loss = 0.
                    valid_result = []
                    valid_label = []

                    for (batch_valid_x, batch_valid_y) in valid_loader:
                        batch_valid_x = batch_valid_x.cuda()
                        batch_valid_y = batch_valid_y.cuda()

                        valid_pred = capsNet(batch_valid_x)
                        b_valid_loss = loss_func(valid_pred, batch_valid_y)
                        valid_loss += float(b_valid_loss)

                        valid_pred = torch.argmax(valid_pred, dim=1)

                        valid_result.extend(valid_pred.detach().cpu().numpy())
                        valid_label.extend(batch_valid_y.detach().cpu().numpy())

                    valid_weighted_accuracy = weighted_accuracy(valid_result, valid_label)
                    valid_unweighted_accuracy = unweighted_accuracy(valid_result, valid_label)
                    valid_confusion = confusion(valid_label, valid_result,
                                                labels=list(range(4)))

                    valid_loss /= len(valid_loader)

                    print('------------Epoch: {:0>3}'.format(epoch),
                          '| cross_validation loss: {:.2f}'.format(valid_loss),
                          '| cross_validation weighted accuracy: {:.2%}'.format(valid_weighted_accuracy),
                          '| cross_validation unweighted accuracy: {:.2%}'.format(valid_unweighted_accuracy),
                          '\nvalid confusion matrix:\n', valid_confusion)

                capsNet.train()

        torch.cuda.empty_cache()


def test(capsNet, loss_func, test_loader):
    test_loss = 0.
    test_result = []
    test_label = []
    capsNet.eval()
    with torch.no_grad():
        for batch_test_x, batch_test_y in test_loader:
            batch_test_x = batch_test_x.cuda()
            batch_test_y = batch_test_y.cuda()

            test_pred = capsNet(batch_test_x)
            b_test_loss = loss_func(test_pred, batch_test_y)
            test_loss += float(b_test_loss)

            test_pred = torch.argmax(test_pred, dim=1)

            test_result.extend(test_pred.detach().cpu().numpy())
            test_label.extend(batch_test_y.detach().cpu().numpy())

        test_weighted_accuracy = weighted_accuracy(test_result, test_label)
        test_unweighted_accuracy = unweighted_accuracy(test_result, test_label)
        test_confusion = confusion(test_label, test_result,
                                    labels=list(range(4)))

        test_loss /= len(test_loader)

        print('------------test loss: {:.2f}'.format(test_loss),
              '| test weighted accuracy: {:.2%}'.format(test_weighted_accuracy),
              '| test unweighted accuracy: {:.2%}'.format(test_unweighted_accuracy),
              '\ntest confusion matrix:\n', test_confusion)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])

    return np.mean(classes_accuracies)


def weighted_accuracy(y_true, y_pred):
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)


if __name__ == '__main__':
    start = time.time()
    capsNet = CapsuleNet()
    capsNet.cuda()
    summary(capsNet, (3, 300, 40))

    print('# parameters:', sum(param.numel() for param in capsNet.parameters()))

    optimizer = torch.optim.Adam(capsNet.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    train_loader, valid_loader, test_loader = load_data('../data/impro_IEMOCAP.pkl')

    train(capsNet, optimizer, loss_func, train_loader, valid_loader)

    test(capsNet, loss_func, test_loader)

    end = time.time()
    print('总用时：{:.2f}mins'.format((end - start) / 60))
