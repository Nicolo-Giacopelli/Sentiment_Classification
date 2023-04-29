


# Imports

import re
import time
import copy
import numpy as np
import pandas as pd
from typing import List
from tqdm.auto import tqdm
from collections import defaultdict, Counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, DataCollatorWithPadding

import warnings  
warnings.simplefilter(action='ignore', category=FutureWarning) 


# Global variables

gen = re.compile("GENERAL")
misc = re.compile("MISCELLANEOUS")    
food = re.compile("FOOD")
first = re.compile("^(\w+)\#")
second = re.compile("^\w+\#(\w+)")
map_sentiment = {'negative': 0, 'neutral': 1, 'positive': 2}



def load_data(train_path, dev_path=None):
  """
  Loads the train and development datasets from the specified file paths and returns them as a dictionary with keys 
  'train' and 'dev'.
  
  Args:
      train_path (str): File path of the training dataset.
      dev_path (str): File path of the development dataset. Defaults to None, when a single subset is being loaded
  
  Returns:
      dataset (DatasetDict): A HuggingFace DatasetDict containing the loaded train and development (if any) datasets
  """
  data_files = {"train": train_path, "dev": dev_path} if dev_path else {"test": train_path}
  
  dataset = load_dataset(
      "csv", 
      data_files=data_files, 
      sep="\t", 
      header=None, 
      names=["sentiment", "aspect", "keyword", "time", "sentence"]
  )
  
  dataset = dataset.remove_columns(["time"])
  
  return dataset



def preprocess_data(dataset, tokenizer, max_len=512, test=False):
    """
    Preprocesses the train and development datasets using the specified tokenizer and returns them.
    The preprocessing consists in formulation of available information with the structure of a question,
    tokenization and renaming of the labels.
    
    Args:
        dataset (DatasetDict): A HuggingFace DatasetDict containing the train and development datasets.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the data.
        max_len (int): length of truncation (not used)
        test (bool): whether to return a single test dataset or train and dev
    
    Returns:
        train_ds (Dataset): The preprocessed training dataset.
        dev_ds (Dataset): The preprocessed development dataset (if any).
    """

    global gen, misc, food, first, second, map_sentiment

    def tokenize_data(df):
       return tokenizer(df["question"], df["sentence"],
                        truncation=True,
                        add_special_tokens=True, 
                        padding=False, 
                        return_attention_mask=True,
                        max_length = max_len,
                        return_token_type_ids=True
                        )

    # Useful strings
    vis = " in general "
    bas = " of "
    base = " of the "
    tem = f"What do you think{vis}{base}"
    tem2 = f"What do you think{base}"
   
    # Express info as a question
    dataset = dataset.map(lambda x: {"question": tem + first.match(x["aspect"]).group(1).lower() + base + x["keyword"] + " ?" if gen.search(x["aspect"]) or misc.search(x["aspect"]) else 
                
                (tem2 + second.match(x["aspect"]).group(1).lower() + bas + first.match(x["aspect"]).group(1).lower() + base + x["keyword"] + " ?" if food.search(x["aspect"]) else 
                 tem2 + second.match(x["aspect"]).group(1).lower() + base + first.match(x["aspect"]).group(1).lower() + base + x["keyword"] + " ?")        
                                    })
    
    # Tokenization
    dataset = dataset.map(tokenize_data)

    # Converting labels from strings to indices
    dataset = dataset.map(lambda x: {'labels': map_sentiment[x['sentiment']]}, remove_columns=['question', 'sentiment', 'aspect', 'keyword', 'sentence'])

    if test:
        return dataset["test"]
    
    return dataset["train"], dataset["dev"]



def compute_accuracy(preds, labels):
  """
  Function to compute accuracy in torch
  """
  # Getting the predicted classes
  preds = torch.argmax(preds, dim=1)

  # Computing the number of correct predictions
  correct = torch.sum(preds.view(-1) == labels.view(-1))

  # Computing the accuracy
  accuracy = correct.item() / len(labels)

  return accuracy



def get_dataloaders(tokenizer, ds, batch_size, shuffle=True):
  '''
  Returns train and dev dataloaders.

  Args:
  - tokenizer (PreTrainedTokenizer): an instance of a Hugging Face tokenizer used for tokenizing input data.
  - df (Dataset): a Hugging Face dataset object containing the preprocessed data.
  - batch size (int): batch size
  - shuffle (bool): whether to shuffle or not the data. Defaults to True.

  Returns:
  - dataloader (DataLoader): A PyTorch DataLoader object for the data.
  '''

  data_collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')
  dataloader = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, collate_fn=data_collator)

  return dataloader
  
  
  
def compute_class_weight(train_ds, device):
  '''
  Computes class weights with which loss is calculated

  Args:
  - train_ds (Dataset): a dictionary containing the training data and labels.
  - device (torch.device): the device on which the tensor is returned

  Returns:
  - weights: A tensor of class weights for the input dataset.
  '''
  # Counting the frequency of each class in the training data
  train_dist = defaultdict(int)
  for i in train_ds['labels']:
    train_dist[i] += 1
  
  # Computing the class weights
  weights = torch.zeros(3)
  total = sum(train_dist.values())

  for idx, _ in enumerate(weights):
    weights[idx] = (train_dist[idx] / total)
  
  return weights.to(device)



def get_model(hf_model_selection: str):
    """
    Loads the specified Hugging Face model and its associated tokenizer and returns them along with the model's configuration.
    
    Args:
        hf_model_selection (str): The name of the HuggingFace model to load.
    
    Returns:
        config (PretrainedConfig): The configuration of the loaded model.
        tokenizer (PreTrainedTokenizer): The tokenizer of the loaded model.
        model (PreTrainedModel): The loaded model.
    """
  
    torch.cuda.empty_cache()
    config = AutoConfig.from_pretrained(hf_model_selection, force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_selection, force_download=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_selection, num_labels=3, output_attentions = True, force_download=True)
    return config, tokenizer, model




class EarlyStopping:
    """ Class used to store the best model checkpoint during training """

    def __init__(self, patience=5, delta=0):
        """
        patience (int): How long to wait after last time validation loss improved. 
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None
        self.best_params = {}

        
    def __call__(self, val_metric, model, save=True):
        """ 
        The validation metric is passed at each iteration, the class keeps track of the best checkpoint
        """

        score = val_metric

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_metric
            self.counter = 0
            if save:
                self.best_model = copy.deepcopy(model).cpu()
                self.best_params = self.best_model.state_dict()
  
  