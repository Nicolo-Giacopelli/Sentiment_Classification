
# NLP_Project - Aspect Term Polarity Classification
### Authors: Leonardo Basili, Jackson Burke, Nicolò Giacopelli

## Introduction
The aim of this project is to implement a classifier that is able to predict opinion polarities of given aspect terms in sentences. That is whether a specific part of the sentence represents a positive, negative or neutral opinion.

## Model instance and model class
As a model instance, we used the large version of RoBERTa, with a vocabulary composed of 50265 tokens. The model is pre-trained on an English corpus (Wikipedia, BookCorpus...), with a masking procedure that is done dynamically during 
pretraining (vs BERT) and tokens that are either masked or replaced by a random token. The model is in fact the same as the base version of Roberta, but pretrained on a larger dataset.
As a model class, with used ModelForSequenceClassification from HugginFace, which returns the model with an additional classification head built on top. 
The classification head takes as input the contextual embedding of the initial token [s] (equivalent to [CLS] for BERT), as it's normally done in classification tasks, and pass it through a dense layer (of the same dimension as the embeddings, 
that is 1024), a Tanh activation and a final connected layer (using Dropout).

## Inputs to the model
As input to the model, we formulated the available information as a question, following the style of question answering from https://aclanthology.org/N19-1035.pdf  (Sun et al., 2019), for instance "What do you think of the quality of
food of the appetizers of olives?" or "what do you think in general of the ambience of the trattoria?". This allows us to use most of the available information in a way that is comprehensible to the model. For each review, we then concatenated
the question and the sentence exploiting that is allowed by the model during pre-training, that is concatenating two consecutive sentence. Contrary to BERT, however, next sentence prediction is not part of the pre-training for the language model
(but still works better). For tokenization, truncation is not performed since all sentences are relatively short in size with respect to those with which the pretraining has been done. The distribution of sentence length is in fact similar for both 
training and development set and peaks around 60 words per sentence, stretching up to 350 (RoBERTa-large is trained on sentence made of 514 words).
When creating the data loaders, we used dynamic padding from DataCollatorWithPadding as a collator, which provides a good compromise between training efficiency and 

## Mixed Precision Training
To speed up training, we exploited Mixed Precision Training from https://pytorch.org/docs/stable/notes/amp_examples.html, which is a numerical format with lower precision than 32-bit floating point to which all variables related to the models are
mapped. This procedure helps with computational workload, requires less memory and facilitates the training of larger networks. In particular, we used the Gradient Scaler technique and final precision up to 16-bit float. 

# Best model checkpoint
During training, we used a custom defined class to save the best model checkpoint, and then reload it after training to predict on the test dataset. The metric that has been tracked to define the best model checkpoint is the accuracy on the development
dataset. This has been done since a weighted CrossEntropy loss has been used, with weights for the mean computed as class frequency from the existing observations (which has proved to work best).

## Extensive parameters tuning
We performed model selection on the development dataset by extensively sweeping over different hyper parameters to finetune the model on the aspect-term polarity analysis task. 
This has been mostly done on Google Collab using a Tesla T4 GPU. Here we share some insights we reached when following this procedure:

- Using a warmup phase seems to improve the performance pre-trained of RoBERTa Large indeed we chose to use "constant_with_warmup" scheduler. During the warmup period, the learning rate increases linearly between 0 and the rate passed as parameter.
  this helps avoiding drastic changes to model parameters that may hinder the convergence of the fine-tuning task.
- The best number of epochs seems to be between 15 and 20 (we got best result of 92.88 accuracy with 20 epochs) however we chose to use 10 epochs since there was a great reduction of computing power and execution time without penalising much the accuracy level
- The optimal learning rate is between 2e-5 and 2.3e-5 much lower or greater learning rates seems to decrease the performance of the model on this task indeed using a learning rate of 2e-4 the accuracy level remain stuck to 0.7031 in all the epochs
- The batch size set to 48 seems to be the best one to optimize the trade off execution time vs accuracy level of the model

# Results
We reported a mean accuracy of 91.44% and a standard deviation equal to 0.68 in 5 runs, as computed on the development dataset (single values: [92.02, 91.49, 90.96, 92.29, 90.43])
The total computational time we reported during these 5 runs is of 1633.68 s (27 minutes), with a single lasting 326 seconds on average.

