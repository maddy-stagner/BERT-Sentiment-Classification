# BERT-Sentiment-Classification
This project fine-tunes a BERT model (bert-base-uncased) for binary sentiment classification (positive or negative) on a dataset of tweets. The implementation uses the Hugging Face transformers library and PyTorch to preprocess data, train the model, and evaluate its performance. The project is structured for reproducibility and modularity, with scripts and configuration files to streamline the workflow.

Project Overview





Objective: Develop a sentiment classification model to predict whether a tweet expresses a positive or negative sentiment.



Model: BERT (bert-base-uncased) fine-tuned for binary classification.



Dataset: A sample of 5,000 tweets from the Sentiment140 dataset, split into 80% training and 20% testing sets.



Performance: Achieves a test accuracy of 78.4% with balanced precision, recall, and F1-scores for both classes.

Dataset

The dataset is a subset of the Sentiment140 dataset, which contains 1.6 million tweets labeled for sentiment:





Labels: 0 (negative), 4 (positive), mapped to 0 and 1, respectively, for binary classification.



Sample Size: 5,000 tweets randomly sampled to reduce computational requirements.



Columns Used: sentiment (label), text (tweet content).



Preprocessing: Tweets are tokenized using BertTokenizer, with a maximum length of 32 tokens, padded or truncated as needed.

Model and Training





Model: BertForSequenceClassification from bert-base-uncased, configured for binary classification (2 labels).



Training Setup:





Epochs: 5



Batch Size: 16



Optimizer: AdamW with a learning rate of 5e-5



Device: GPU (if available) or CPU



Training Metrics:





Epoch 1: Loss = 0.522, Accuracy = 0.748



Epoch 2: Loss = 0.299, Accuracy = 0.878



Epoch 3: Loss = 0.142, Accuracy = 0.951



Epoch 4: Loss = 0.078, Accuracy = 0.977



Epoch 5: Loss = 0.066, Accuracy = 0.979

Performance

The model was evaluated on a test set of 1,000 tweets (20% of the sampled data). The performance metrics are:





Test Accuracy: 78.4%
