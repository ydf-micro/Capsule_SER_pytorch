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

np.set_printoptions(formatter={'float': '{:6.2%}'.format})


def get_percentage(confusion):
    # get the percentage of confusion matrix values
    true_class_num = np.expand_dims(np.sum(confusion, axis=1), axis=-1)

    return confusion / true_class_num


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = torch.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)

    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


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
                 kernel_size, strides, padding):
        super(CapsuleLayer, self).__init__()

        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.capsules = nn.ModuleList(   # 8*36*5*8
            [nn.Conv2d(self.in_channels, self.out_channels,
                       kernel_size=self.kernel_size,
                       stride=self.strides, padding=padding)
             for _ in range(self.num_capsules)]
        )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)

        return scale * tensor / torch.sqrt(squared_norm + eps)

    def forward(self, x):
        # [batch_size, 8*36*5, 1] * 8
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
        # x = [batch_size, 8*38*5, 8]
        # self.rout_weights = [4, 8*38*5, 8, 16]
        # priors: return [4, batch_size, 8*38*5, 1, 16]
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
                      out_channels=32,
                      kernel_size=(7, 1),
                      stride=1,
                      padding=(3, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3, 1),
                      stride=1,
                      padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(1, 5),
                      stride=1,
                      padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.blstm = BLSTM(
            input_size=20 * 128,
            hidden_size=128,
            num_layers=2,
            num_classes=4
        )

        self.primaryCapsules = CapsuleLayer(
            num_capsules=8,
            in_channels=256,
            out_channels=8,
            kernel_size=(5, 3),
            strides=2,
            padding=(2, 1)
        )

        self.digitCapsules = DigitCapsules(
            num_capsules=NUM_CLASS,
            num_route_nodes=8*38*5,
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
                                            to_lstm.size(2) * to_lstm.size(3))
        to_lstm = self.blstm(to_lstm)

        to_caps = self.primaryCapsules(x)
        to_caps = self.digitCapsules(to_caps)

        to_caps = torch.sqrt(torch.sum(torch.pow(to_caps, 2), dim=-1))

        # print(to_lstm)
        # print(to_caps)

        out = to_lstm + to_caps
        # print(out)

        return out


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


def train(capsNet, optimizer, scheduler, loss_func, train_loader, valid_loader):

    for epoch in range(1, EPOCHS + 1):

        train_loss = 0.
        train_result = []
        train_label = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            pred = capsNet(batch_x)
            loss = loss_func(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(pred, dim=1)

            train_result.extend(pred.cpu())
            train_label.extend(batch_y.cpu())
            train_loss += float(loss)

            train_accuracy = (pred==batch_y).float().mean()

            print(f'Epoch: {epoch:0>3}/{EPOCHS}',
                  f'| step: {step+1:0>2}/{len(train_loader)}',
                  f'| loss: {loss.item():.2f}',
                  f'| train accuracy: {train_accuracy.item():.2%}')

            if step == len(train_loader) - 1:

                train_result = torch.tensor(train_result)
                train_label = torch.tensor(train_label)

                train_weighted_accuracy = weighted_accuracy(train_result, train_label)
                train_unweighted_accuracy = unweighted_accuracy(train_result, train_label)
                train_confusion = confusion(train_label, train_result,
                                            labels=list(range(4)))
                train_confusion = get_percentage(train_confusion)

                train_loss /= len(train_loader)

                print(f'------------Epoch: {epoch:0>3}/{EPOCHS}',
                      f'| train loss: {train_loss:.2f}',
                      f'| train weighted accuracy: {train_weighted_accuracy.item():.2%}',
                      f'| train unweighted accuracy: {train_unweighted_accuracy.item():.2%}',
                      '\ntrain confusion matrix:[hap, sad, ang, neu]\n', train_confusion)

            torch.cuda.empty_cache()


        if epoch % 10 == 0:
            torch.save(capsNet, f'../model/model_{epoch:0>2}.pkl')
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

                    valid_result.extend(valid_pred.cpu())
                    valid_label.extend(batch_valid_y.cpu())

                valid_result = torch.tensor(valid_result)
                valid_label = torch.tensor(valid_label)

                valid_weighted_accuracy = weighted_accuracy(valid_result, valid_label)
                valid_unweighted_accuracy = unweighted_accuracy(valid_result, valid_label)
                valid_confusion = confusion(valid_label, valid_result,
                                            labels=list(range(4)))
                valid_confusion = get_percentage(valid_confusion)

                valid_loss /= len(valid_loader)

                print(f'------------Epoch: {epoch:0>3}',
                      f'| cross_validation loss: {valid_loss:.2f}',
                      f'| cross_validation weighted accuracy: {valid_weighted_accuracy.item():.2%}',
                      f'| cross_validation unweighted accuracy: {valid_unweighted_accuracy.item():.2%}',
                      '\nvalid confusion matrix:[hap, sad, ang, neu]\n', valid_confusion)

            capsNet.train()

            torch.cuda.empty_cache()

        scheduler.step()


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

            test_result.extend(test_pred.cpu())
            test_label.extend(batch_test_y.cpu())

        test_result = torch.tensor(test_result)
        test_label = torch.tensor(test_label)

        test_weighted_accuracy = weighted_accuracy(test_result, test_label)
        test_unweighted_accuracy = unweighted_accuracy(test_result, test_label)
        test_confusion = confusion(test_label, test_result,
                                    labels=list(range(4)))
        test_confusion = get_percentage(test_confusion)

        test_loss /= len(test_loader)

        print(f'------------test loss: {test_loss:.2f}',
              f'| test weighted accuracy: {test_weighted_accuracy.item():.2%}',
              f'| test unweighted accuracy: {test_unweighted_accuracy.item():.2%}',
              '\ntest confusion matrix:[hap, sad, ang, neu]\n', test_confusion)


def unweighted_accuracy(y_true, y_pred):
    classes = torch.unique(y_true)
    classes_accuracies = torch.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])

    return classes_accuracies.mean()


def weighted_accuracy(y_true, y_pred):

    return (y_true == y_pred).float().mean()


if __name__ == '__main__':
    start = time.time()
    capsNet = CapsuleNet()
    print(capsNet)
    capsNet.cuda()
    summary(capsNet, (3, 300, 40))

    print('# parameters:', sum(param.numel() for param in capsNet.parameters()))

    # Use L2 normalization
    weight_p, bias_p = [], []
    for name, p in capsNet.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    optimizer = torch.optim.Adam([
        {'params': weight_p, 'weight_decay': 0.01},
        {'params': bias_p, 'weight_decay': 0}
        ], lr=LR)

    optimizer = torch.optim.Adam(capsNet.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[180], gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    train_loader, valid_loader, test_loader = load_data('../data/impro_IEMOCAP.pkl')

    train(capsNet, optimizer, scheduler, loss_func, train_loader, valid_loader)

    test(capsNet, loss_func, test_loader)

    end = time.time()
    print(f'总用时：{(end - start) / 60:.2f}mins')

    # for i in range(10, 210, 10):
    #     capsNet = torch.load(f'../model/model_{i}.pkl')
    #
    #     print(f'-------------------------------Epoch: {i}')
    #
    #     test_model(capsNet, test_loader)