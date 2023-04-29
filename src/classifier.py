

from functions import * 


# Global variables

inverse_map_sentiment = {0: "negative", 1: "neutral", 2: "positive"}
train_batch_size = 48
val_batch_size = 48
test_batch_size = 48


# Best hyperparameters we found in our model selection (hardcoded)

lr =  2.2e-5   # learning rate
wd = 0.001   # weight decay
class_balance = True  # loss weights
sched_name = "constant_with_warmup" 
warmup_ratio = 0.15   # for scheduler
epochs = 10  # number of epochs


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
     
    def __init__(self):
      self.config, self.tokenizer, self.model = get_model('roberta-large')
      self.early_stopping = EarlyStopping(patience=float("inf"), delta=0)
      self.model_fin = None  # best model with which we predict
          
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        
        global train_batch_size, val_batch_size, lr, wd, class_balance, sched_name, warmup_ratio, epochs
        
        dataset = load_data(train_filename, dev_filename)
        train_ds, dev_ds = preprocess_data(dataset, self.tokenizer, test=False)
        
        # Preprocessing the input data
        train_dataloader = get_dataloaders(self.tokenizer, train_ds, train_batch_size, True)
        dev_dataloader = get_dataloaders(self.tokenizer, dev_ds, val_batch_size, False)
      
        # Setting up the optimizer and loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        weight = compute_class_weight(train_ds, device) if class_balance else None  # None
        criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        
        # Setting the device for training
        self.model.to(device)
        
        # Setting up the learning rate scheduler
        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(name=sched_name, optimizer=optimizer, num_warmup_steps=num_training_steps*warmup_ratio, num_training_steps=num_training_steps)
        
        # Starting loop
        for _ in range(epochs):
          scaler = GradScaler()
          
          ## TRAIN
          self.model.train()
          for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
              outputs = self.model(**batch)
              # Loss
              logits = outputs.logits
              tmp_train_loss = criterion(logits.view(-1, 3), batch['labels'].long().view(-1))

            scaler.scale(tmp_train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            torch.cuda.empty_cache()
            
          eval_accuracy = 0

          ## DEV
          self.model.eval()
          for batch in dev_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
              outputs = self.model(**batch)
              logits = outputs.logits.detach()
              
              # Accuracy
              label_ids = batch['labels']
              eval_accuracy += compute_accuracy(logits, label_ids)
          
          eval_accuracy = eval_accuracy / len(dev_dataloader)
          self.early_stopping(eval_accuracy, self.model, save=True)
         
        # Reload Best Model from early stopper
        self.model_fin = self.early_stopping.best_model
        

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        
        global test_batch_size, lr, wd, class_balance, sched_name, warmup_ratio, epochs, inverse_map_sentiment
        
        dataset = load_data(data_filename)
        test_ds = preprocess_data(dataset, self.tokenizer, test=True)
        
        # Preprocessing the input data
        test_dataloader = get_dataloaders(self.tokenizer, test_ds, test_batch_size, False)
        
        
        # Setting the device for training
        self.model_fin.to(device)
        self.model_fin.eval()
        
        y_pred = []
        
        ## TEST
        for i, batch in enumerate(test_dataloader):
          batch = {k: v.to(device) for k, v in batch.items()}
          with torch.no_grad():
            outputs = self.model_fin(**batch)
            # Loss
            logits = outputs.logits.detach().cpu().numpy()
            updated = [inverse_map_sentiment[key] for key in logits.argmax(axis=1).tolist()]
            y_pred.extend(updated)
        
        return y_pred
    


