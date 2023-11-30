import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from typing import Optional

def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    This function implements the core components of any Neural Network training regiment.
    In our stochastic setting our code follows a very specific "path". First, we load the batch
    a single batch and zero the optimizer. Then we perform the forward pass, compute the gradients and perform the backward pass. And ...repeat!
    """

    running_loss = 0.0
    model = model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # move data and target to device
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # do the forward pass
        output = model(data)

        # compute the loss
        loss = F.cross_entropy(output, target)

        # compute the gradients
        loss.backward()

        # perform the gradient step
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    return running_loss / len(train_dataloader)

def fit(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    valid_dataloader: Optional[DataLoader]=None,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None):
    '''
    the fit method simply calls the train_epoch() method for a
    specified number of epochs.
    '''

    # keep track of the losses in order to visualize them later
    # Train for numerous epochs:
    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(epochs):
        train_loss = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device
        )
        train_losses.append(train_loss)

        if valid_dataloader is not None:
            valid_loss, valid_acc = predict(model, valid_dataloader, device, verbose=False)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
        if scheduler is not None:
          scheduler.step(train_loss)
        if valid_dataloader is None:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Validation Loss={valid_loss:.4f}, Validation acc={valid_acc:.4f}")
    return train_losses, valid_losses, valid_accs


def predict(
    model: nn.Module, test_dataloader: DataLoader, device: torch.device, verbose=True
):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="sum")
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_dataloader.dataset)
    accuracy = 100.0 * correct / len(test_dataloader.dataset)

    if verbose:
        print(
            f"Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({accuracy:.0f}%)"
        )

    return test_loss, accuracy


def visualize_images(dataloader):
    images = next(iter(dataloader))[0][:10]
    grid = torchvision.utils.make_grid(images, nrow=5, padding=10)

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")

    show(grid)

def test_models(models,optimizers,train_dataloader,valid_dataloader,test_dataloader,DEVICE,scheduler=None):
    # initialize model
    train_losses_all = []
    validate_losses_all = []
    valid_accs_all = []
    test_loss_all = []
    test_acc_all = []

    for i, (optimizer,model) in enumerate(zip(optimizers,models)):
      # train the MLP
      print(f'****** Model {i+1} **************')
      train_losses, valid_losses, valid_accs = fit(
          model=model,
          train_dataloader=train_dataloader,
          valid_dataloader=valid_dataloader,
          optimizer=optimizer,
          epochs=20,
          device=DEVICE,
          scheduler=scheduler
      )
      train_losses_all.append(train_losses)
      validate_losses_all.append(valid_losses)
      valid_accs_all.append(valid_accs_all)
      test_loss, test_accuracy = predict(model=model, test_dataloader=test_dataloader, device=DEVICE)
      test_loss_all.append(test_loss)
      test_acc_all.append(test_accuracy)
    return train_losses_all, validate_losses_all, valid_accs_all, test_loss_all, test_acc_all

def plot_losses(train_losses_all,validate_losses_all,all_in_one=True):
  for i, (train_loss, validate_loss) in enumerate(zip(train_losses_all, validate_losses_all)):
    plt.plot(train_loss, '-*',label = 'train, model ' + str(i+1))
    plt.plot(validate_loss,'-*', label = 'validate, model '+str(i+1))
    if not all_in_one:
      plt.title("Loss progression across epochs")
      plt.xlabel("Epochs")
      plt.ylabel("Loss")
      plt.legend()
      plt.semilogy()
      plt.show()
  if all_in_one:
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.semilogy()
    plt.title("Loss progression across epochs")
    plt.show()
