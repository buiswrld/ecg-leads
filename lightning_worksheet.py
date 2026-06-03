"""
Goal:
Connect the dataset, dataloaders, model, loss function, optimizer, and trainer.

This file is intentionally incomplete. Find active tasks marked by #TODO and fill in the blanks below them.
Some parts may be omitted or added depending on necessity. You can add more code if you think it is necessary.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# * Import your dataset class.
from dataset import PTBXLDataset 
# i guess the file for the dataset should be dataset.py

# Here, we also need to import a ResNet class, but we leave this exercise for another time, temporarily.
# from resnet1d import ResNet1D


# ------------------------------------------------------------
# Basic settings
# ------------------------------------------------------------

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# This folder contains the array with the waveforms and the preprocessed .csv
data_path = "/content/drive/MyDrive/ptbxl_project/"

PARAMS = {
    # * Fill in the paths corresponding to where the CSV and data are located for our dataset.
    # Since we are using Colab, make sure to mount your Google Drive at the top and update the paths accordingly.
    "csv_path": data_path + "processed_ptbxl_metadata.csv",
    "data_root": data_path,

    # None means use all 12 ECG leads.
    # Later, we can replace this with a smaller list of selected leads.
    "leads": None,

    "batch_size": 32,
    "num_workers": 4,
    "learning_rate": 1e-3,
    "max_epochs": 10,

    # PTB-XL diagnostic superclass labels:
    # NORM, MI, STTC, CD, HYP
    "num_classes": 5,
}


# ------------------------------------------------------------
# Training task
# ------------------------------------------------------------

class ECGClassificationTask(pl.LightningModule):
    """
    Defines how the model is trained, validated, and tested.

    This section should include:
    - the model
      - This is where you will define the architecture of your model.
      - We will be using ResNet, but for now, leave this as a placeholder.

    - the loss function
      - This is where you will define the loss function used to train the model.
      - The "loss function" is another way to say "objective function", the function we want to minimize to make the model better.
      - For a multi-label classification problem, this is typically binary cross-entropy loss.
      - In PyTorch, BCEWithLogitsLoss is usually the correct choice when the model outputs raw logits.

    - the training step
      - This is where you will define what happens during one step of training.
      - Training is the process of updating the model's parameters to minimize the loss on the training data.
      - This typically includes getting a batch of data, running it through the model, computing the loss, and logging the loss.

    - the validation step
      - This is where you will define what happens during one step of validation.
      - Validation is used to check how the model is doing on unseen data during training, and to tune hyperparameters.
      - It is not used to evaluate the final model.
      - This typically includes getting a batch of data, running it through the model, computing the loss, and logging the loss.

    - the test step
      - This is where you will define what happens during one step of testing.
      - Testing is similar to validation, but it is used to evaluate the final model after training is complete.
      - This typically includes getting a batch of data, running it through the model, computing the loss, and logging the loss.

    - the optimizer
      - This is where you will define which optimizer to use for training the model.
      - An optimizer's job is to update the model's parameters based on the computed gradients to minimize the loss.
      - This typically includes creating an optimizer, like Adam or SGD, and telling Lightning to use it.
    """

    def __init__(self, num_leads, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the model.
        # # For now, leave this as a placeholder. Later, we will replace this with a ResNet architecture, which we will define ourselves in another file. This file will not train until you replace None with a real model.
        self.model = None

        # TODO: Initialize the loss function.
        # 1. What do you think is a good loss function for multi-label classification?
        # 2. Should the loss expect probabilities or raw logits?
        self.loss_fn = None

    def forward(self, ecg):
        """
        Goal:
        Define how the model produces output from input ECGs.
        """

        # TODO: Run the ECG through the model.
        # Call self.model on the input ECG to get the model's output.
        # 1. What do you think the shape of the output should be?
        # 2. Should you apply an activation function here?
        # Hint: Check what BCEWithLogitsLoss expects as input.

        # Returns: raw logits of shape (batch_size, num_classes).
        # # Note/Hint: Do not apply sigmoid here if using BCEWithLogitsLoss. Sigmoid can be used later when converting logits into probabilities for evaluation or interpretation.
        pass

    def training_step(self, batch, batch_idx):
        """
        Goal:
        Use one batch to compute training loss.
        """

        # TODO: Get ECGs and labels from the batch.
        # 1. How can you get the ECGs and labels from the batch?
        # 2. What do you think their shapes will be? Can we check?

        # TODO: Get model outputs.
        # Call self.forward on the input ECGs to get the model's raw logits.

        # TODO: Compute loss.
        # Call the loss function on the model's logits and the true labels to compute the loss.

        # TODO: Log training loss.
        # Use self.log to log the training loss. This will allow us to log it and track it over time.

        # TODO: Return loss.
        # The training step should return the loss, which will be used by the optimizer to update the model's parameters.

        # Returns: the training loss.
        pass

    def validation_step(self, batch, batch_idx):
        """
        Goal:
        Check how the model performs on validation data.
        """

        # TODO: Get ECGs and labels from the batch.
        # 1. How can you get the ECGs and labels from the batch?
        # 2. What do you think their shapes will be? Can we check?

        # TODO: Get model outputs.
        # Call self.forward on the input ECGs to get the model's raw logits.

        # TODO: Compute validation loss.
        # Call the loss function on the model's logits and the true labels to compute the loss.

        # TODO: Log validation loss.
        # Use self.log to log the validation loss. This will allow us to log it and track it over time.

        # Returns: the validation loss.
        pass

    def test_step(self, batch, batch_idx):
        """
        Goal:
        Evaluate the final model on test data.
        """

        # TODO: Write a test step.
        # This will be similar to the validation step, but it will be used to evaluate the final model after training is complete.
        # See the validation step for guidance.

        # Returns: the test loss.
        pass

    def configure_optimizers(self):
        """
        Goal:
        Tell Lightning which optimizer to use.
        """

        # TODO: Create and return an optimizer.
        # 1. Which optimizer do you think is a good choice for training this model? Why?
        # 2. How do you create an optimizer in PyTorch?
        # 3. What parameters do you need to pass to it?

        # Returns: the optimizer.
        pass


# ------------------------------------------------------------
# Data setup
# ------------------------------------------------------------

def build_dataloaders(params):
    """
    Goal:
    Create train, validation, and test dataloaders.

    The dataset handles loading ECGs and labels.
    The dataloader handles batching and shuffling.

    Important:
    The training dataset should create or fit the label binarizer.
    The validation and test datasets should reuse the same label binarizer.
    This keeps the label order consistent across train, validation, and test.
    """

    # * Create the training dataset.
    # 1. What parameters do you need to create the dataset?
    # 2. Check the PTBXLDataset __init__ method to see what arguments are required.
    train_dataset = PTBXLDataset(csv_path=params["csv_path"], data_root=params["data_root"], split="train", leads=params["leads"])

    # * Create the validation dataset.
    # See the training dataset for guidance.
    # This will be similar, but it will use the validation data instead of the training data.
    # Make sure it uses the same label binarizer as the training dataset.
    val_dataset = PTBXLDataset(csv_path=params["csv_path"], data_root=params["data_root"], split="val", leads=params["leads"], mlb=train_dataset.mlb)

    # * Create the test dataset.
    # See the training dataset for guidance.
    # This will be similar, but it will use the test data instead of the training data.
    # Make sure it uses the same label binarizer as the training dataset.
    test_dataset = PTBXLDataset(csv_path=params["csv_path"], data_root=params["data_root"], split="test", leads=params["leads"], mlb=train_dataset.mlb)

    # TODO: Create the training dataloader.
    # 1. What parameters do you need to create the dataloader?
    # 2. Check the DataLoader documentation or examples to decide which arguments are needed.
    # 3. Do we want to shuffle the training data? What about the validation and test data? Why or why not?
    train_loader = None

    # TODO: Create the validation dataloader.
    # See the training dataloader for guidance.
    # This will be similar, but it will use the validation dataset instead of the training dataset.
    val_loader = None

    # TODO: Create the test dataloader.
    # See the training dataloader for guidance.
    # This will be similar, but it will use the test dataset instead of the training dataset.
    test_loader = None

    # TODO: Return the dataloaders and anything else needed later.
    # Returns: the training dataloader, the validation dataloader, and the test dataloader.
    pass


# ------------------------------------------------------------
# Lead helper
# ------------------------------------------------------------

def get_num_leads(params):
    """
    Goal:
    Figure out how many ECG leads the model should expect.
    """

    # TODO: Return 12 if using all leads. Otherwise, return the number of selected leads.
    # 1. How can we tell if we are using all leads or not?
    # 2. Is there a parameter we can check in the PARAMS dictionary? If not, how can we modify the PARAMS dictionary to include this information?

    # Returns: the number of leads.
    pass
