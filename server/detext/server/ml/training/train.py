import copy
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Solver:
    def __init__(self, criterion, dataloaders, model, cb=None, verbose=True):
        self.criterion = criterion
        self.model = model

        self.dataloaders = dataloaders
        self.dataset_sizes = {
            "train": len(dataloaders["train"].dataset),
            "test": len(dataloaders["test"].dataset)
        }

        self.verbose = verbose

        self.cb = cb

    def train(self, device='cpu', num_epochs=25, lr=1e-3, step_size=7):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=0.1)

        self.device = device

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        since = time.time()

        for epoch in range(num_epochs):
            self.log('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.log('-' * 10)

            epoch_acc = self.train_epoch()

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
            if self.cb is not None:
                self.cb(epoch, epoch_acc)

        time_elapsed = time.time() - since
        self.log('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.log('Best val Acc: {:4f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)
        return self.model, best_acc

    def train_epoch(self):
        for phase in ['train', 'test']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss, preds = self.iteration(inputs, labels, phase)

                if i % 5 == 0:
                    self.log('  {}/{}: Loss: {:.4f}'.format(
                        i, len(self.dataloaders[phase]), loss.item()))

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                self.scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

            self.log('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        return epoch_acc

    def iteration(self, inputs, labels, phase):
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()
        return loss, preds

    def log(self, msg):
        if self.verbose:
            print(msg)
