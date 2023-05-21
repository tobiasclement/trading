# DQN-Trading

This is a framework based on deep reinforcement learning for stock market trading. This project is the implementation
code for the two papers:

- [Learning financial asset-specific trading rules via deep reinforcement learning](https://arxiv.org/abs/2010.14194)
- [A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules](https://arxiv.org/abs/2101.03867)

with the adaption to be trained on additional index data (one or two).

The deep reinforcement learning algorithm used here is Deep Q-Learning.

## Acknowledgement

- [Deep Q-Learning tutorial in pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## Requirements

Install pytorch using the following commands. This is for CUDA 11.1 and python 3.8:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
- python = 3.8
- pandas = 1.3.2
- numpy = 1.21.2
- matplotlib = 3.4.3
- cython = 0.29.24
- scikit-learn = 0.24.2

## Usage
```bash
python main.py --dataset_name name --index int 
```

## Example AAPL with 2_index data 
```bash
python main.py --dataset_name APPl --index 2
```

## Important
* index allows for 0 (no index data), 1 (one index data), 2 (two index data)
* column names in data set need to look like this:
  Date	Open	High	Low	Close	Adj Close	Volume	Open_1	High_1	Low_1	Close_1	Adj Close_1	Open_2	High_2	Low_2	Close_2	Adj Close_2

* adjust the split_point, begin_date, end_date (dates have to be within dataset) in main.py
* load_from_file= False (data is being preprocessed) in main.py
* e.g. 'AAPL': YahooFinanceD

## References

```
@article{taghian2020learning,
  title={Learning financial asset-specific trading rules via deep reinforcement learning},
  author={Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza},
  journal={arXiv preprint arXiv:2010.14194},
  year={2020}
}

@article{taghian2021reinforcement,
  title={A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules},
  author={Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza},
  journal={arXiv preprint arXiv:2101.03867},
  year={2021}
}
```
