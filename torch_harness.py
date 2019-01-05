""" TorchHarness """
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


def iteration_number(epoch, loader, batch_idx):
    """ calculate iteration number """
    return (epoch * len(loader)) + batch_idx


class TorchHarness:
    """ Harness for the training and validation of models """

    def __init__(self, model, model_name, train, test, epochs=2):
        self.model_name = model_name
        self.all_labels = []
        self.train = train
        self.test = test
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_and_evaluate(self):
        """ train and evaluate model performance """
        learning_rate = 0.0001
        torch.manual_seed(1)

        arch = self.model_name

        time = datetime.now().strftime("{}_%b_%d_%I_%M%S".format(arch))

        writer = SummaryWriter('training_logs/{}'.format(time))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                               weight_decay=0.0,
                               amsgrad=False)

        dl_args = {'batch_size': 32, 'shuffle': True, 'num_workers': 1}

        train_loader = torch.utils.data.DataLoader(dataset=self.train, **dl_args)
        valid_loader = torch.utils.data.DataLoader(dataset=self.test, **dl_args)

        # after we hit > 75% accuracy, we begin saving model checkpoints:
        # current_accuracy = 50.
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(train_loader, optimizer, epoch, writer)
            self.valid_epoch(valid_loader, epoch, writer)

    def train_epoch(self, train_loader, optimizer, epoch, writer):
        """ Train the model """
        embedding_log = 20

        print("Epoch: {}".format(epoch))

        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss.data.item(),
                              iteration_number(epoch, train_loader, batch_idx))

            if batch_idx % embedding_log == 0:
                print("Train/loss: {}".format(loss.data.item()))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
                )

    def valid_epoch(self, test_loader, epoch, writer):
        """ Compare our model with the validation set """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, label = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, label)
                test_loss += loss.item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()
                writer.add_scalar('valid/loss', loss.data.item(),
                                  iteration_number(epoch, test_loader, idx))

        test_loss /= len(test_loader.dataset)

        acc = 100. * correct / len(test_loader.dataset)
        writer.add_scalar('valid/accuracy', acc, (epoch * len(test_loader)))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), acc))

        return acc
