# trading

Deep Reinforcement Learning for Stock Market Trading

This repository contains the implementation of a deep reinforcement learning model for predicting buy, sell, or hold actions in the stock market. The model is trained on historical data and uses a Deep Q-Learning framework to make predictions.

Features
Considers stock data and additionally index data (one or two)
Implements the Deep Q-Learning algorithm for reinforcement learning.
Uses time-series data and convolutional models for state representation.
Includes various encoder-decoder architectures for feature extraction and decision making.
Provides options for batch training and hard update policy.
Allows for a customizable state space, including patterns extracted from technical analysis and Open, High, Low, and Close prices.

In this section, I briefly explain different parts of the project and how to change each. The data for the project downloaded from Yahoo Finance where you can search for a specific market there and download your data under the Historical Data section. Then you create a directory with the name of the stock under the data directory and put the .csv file there.

The DataLoader directory contains files to process the data and interact with the RL agent. The DataLoader.py loads the data given the folder name under Data folder and also the name of the .csv file. For this, you should use the YahooFinanceDataLoader class for using data downloaded from Yahoo Finance.

The Data.py file is the environment that interacts with the RL agent. This file contains all the functionalities used in a standard RL environment. For each agent, I developed a class inherited from the Data class that only differs in the state space (consider that states for LSTM and convolutional models are time-series, while for other models are simple OHLCs). In DataForPatternBasedAgent.py the states are patterns extracted using rule-based methods in technical analysis. In DataAutoPatternExtractionAgent.py states are Open, High, Low, and Close prices (plus some other information about the candle-stick like trend, upper shadow, lower shadow, etc). In DataSequential.py as it is obvious from the name, the state space is time-series which is used in both LSTM and Convolutional models. DataSequencePrediction.py was an idea for feeding states that have been predicted using an LSTM model to the RL agent. This idea is raw and needs to be developed.

Where ever we used encoder-decoder architecture, the decoder is the DQN agent whose neural network is the same across all the models.

The DeepRLAgent directory contains the DQN model without encoder part (VanillaInput) whose data loader corresponds to DataAutoPatternExtractionAgent.py and DataForPatternBasedAgent.py; an encoder-decoder model where the encoder is a 1d convolutional layer added to the decoder which is DQN agent under SimpleCNNEncoder directory; an encoder-decoder model where encoder is a simple MLP model and the decoder is DQN agent under MLPEncoder directory.

Under the EncoderDecoderAgent there exist all the time-series models, including CNN (two-layered 1d CNN as encoder), CNN2D (a single-layered 2d CNN as encoder), CNN-GRU (the encoder is a 1d CNN over input and then a GRU on the output of CNN. The purpose of this model is that CNN extracts features from each candlestick, thenGRU extracts temporal dependency among those extracted features.), CNNAttn (A two-layered 1d CNN with attention layer for putting higher emphasis on specific parts of the features extracted from the time-series data), and a GRU encoder which extracts temporal relations among candles. All of these models use DataSequential.py file as environment.

For running each agent, please refer to the Main.ipynb file for instructions on how to run each agent and feed data. The Main.ipynb file also has code for plotting results.

The Objects directory contains the saved models from our experiments for each agent.

The PatternDetectionCandleStick directory contains Evaluation.py file which has all the evaluation metrics used in the paper. This file receives the actions from the agents and evaluate the result of the strategy offered by each agent. The LabelPatterns.py uses rule-based methods to generate buy or sell signals. Also Extract.py is another file used for detecting wellknown candlestick patterns.

RLAgent directory is the implementation of the traditional RL algorithm SARSA-λ using cython. In order to run that in the Main.ipynb you should first build the cython file. In order to do that, run the following script inside it's directory in terminal:

python setup.py build_ext --inplace
This works for both linux and windows.

For more information on the algorithms and models, please refer to the original paper. You can find them in the references.

Usage
python main.py --dataset_name name --index int

Example
AAPL with 2_index data
python main.py --dataset_name APPl --index 2

Important: adjustments in main.py
- adjust the split_point, begin_date, end_date (dates have to be within dataset)
- load_from_file= False (data is being preprocessed)
- e.g.     'AAPL': YahooFinanceDataLoader(args.index,
                                   'AAPL',
                                   split_point='2020-01-02',
                                   begin_date='2017-01-03',
                                   end_date='2023-05-12',
                                   load_from_file=False,
                                   ),


Contributing
If you have questions about the project or would like to contribute, please send an email to (your email address).

