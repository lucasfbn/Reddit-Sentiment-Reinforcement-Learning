# Reddit Sentiment Stock Movements Prediction

**Note:** This standalone version relies on other components not included in this repository (on purpose). This also applies to the collected data.

## Introduction

In recent years, social media has become an increasingly popular way for people to connect with one another and share information. As a result, sentiment analysis of social media has become a valuable tool for predicting stock movements.

One platform that has been used extensively for sentiment analysis is Reddit. Reddit is a social news aggregator and discussion forum where users can submit, vote, and comment on content. Due to the large amount of user-generated content, Reddit has been found to be a valuable source for gauging public opinion.

In this repository, we will be using reinforcement learning to train two agents to predict the movement of stock prices based on Reddit sentiment. The agents are using either a multimodal convolutional neural network.

## Directory structure

### data

Data related to stock symbols. The dataset (stock prices + scraped Reddit content) is not included.

### dataset_handler

The dataset is comprised of meta files and HDF5 files. The dataset handler merges the files accordingly to form a complete dataset.

### preprocessing

 Preprocessing steps (tasks.py) and pipeline (pipeline.py)

### rl

Reinforcement learning module (using stable-baselines3). This module is further divided into "portfolio", "simulation", and "stocks". Two reinforcement learning agents are trained. The "stocks" submodule trains an agent capable of trading individual stocks. The "portfolio" submodule uses the trained "stocks" agent to trade a portfolio of stocks. The "simulation" submodule is a semi-realistic test environment for the "portfolio" submodule. 

### sentiment_analysis

Sentiment analysis steps (tasks.py) and pipeline (pipeline.py). The sentiment analysis is run before the preprocessing steps (see above). 

```
root
  data                    # Data related to stock symbols
  dataset_handler         # Dataset handler to handle meta files and HDF5 files
  preprocessing           # Preprocessing steps (tasks.py) and pipeline (pipeline.py)
  rl                      # Reinforcement learning module (using stable-baselines3)
    common                # Shared functionality
    portfolio             # Portfolio trading environment + runner
    simulation            # Real market simulation (essentially a test environment)
    stocks                # Individual stock training environment + runner
  sentiment_analysis      # Sentiment analysis steps (tasks.py) and pipeline (pipeline.py)
  tests                  
  utils
```

## Technologies used

GCP, PyTorch, Stable-Baselines3, Skicit-Learn, Weights & Biases, Ray, Optuna, nltk

