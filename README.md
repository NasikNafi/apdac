# Attention-based Partially Decoupled Actor-Critic (APDAC)

This repository contains the code for the following paper presented at the Deep RL Workshop, NeurIPS 2021:\
*[Attention-based Partial Decoupling of Policy and Value for Generalization in Reinforcement Learning](http://idl.iscram.org/files/nasikmuhammadnafi/2020/2279_NasikMuhammadNafi_etal2020.pdf)*.


# Citation
If you use this code, please cite our paper:
```
Nafi, N.M., Glasscock, C. and Hsu, W. (2021). Attention-based Partial Decoupling of Policy and Value for Generalization in Reinforcement Learning. In Deep Reinforcement Learning Workshop, NeurIPS 2021.
```

Our code is largely based on *[this](https://github.com/rraileanu/idaac)* implementation and the corresponding paper is available *[here](https://arxiv.org/abs/2102.10330)*. Their implementation used an open sourced [PyTorch implementation of PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


# Dependencies
Run the following to create the environment and install the required dependencies: 
```
conda create -n apdac python=3.7
conda activate apdac

cd apdac
pip install -r requirements.txt

pip install procgen

git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 
```


# Instructions 

## To Train APDAC on CoinRun
```
python train.py --env_name coinrun --algo apdac
```

## To Train IDAAC on CoinRun
```
python train.py --env_name coinrun --algo idaac
```

## To Train PPO on CoinRun
```
python train.py --env_name coinrun --algo ppo --ppo_epoch 3
```



APDAC uses the same set of hyperparameters for all environments. Please refer to the paper for the details and the experimental results. **APDAC** significantly outperforms the PPO baseline and achieves comparable performance with respect to the recent state-of-the-art method IDAAC on the challenging RL generalization benchmark [Procgen](https://openai.com/blog/procgen-benchmark/). Thus, APDAC demonstrates similar generalization benefits of a fully decoupled approach while reducing the overall parameters and
computational cost.
