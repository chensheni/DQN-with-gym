# DQN-with-gym

This is the flappy bird game:
[https://github.com/markub3327/flappy-bird-gymnasium](https://github.com/robertoschiavone/flappy-bird-env)

1. create vitual environment
```
conda create -n dqnenv
```
2. check v env
```
conda info -e
```
3. activate the v env
```
conda activate dqnenv
```
4. install python
```
conda install python=3.11
```
5. install flappy-bird-gymnasium
```
pip install flappy-bird-gymnasium
```
6. install pytorch within the v env
```
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```
7. train models
```
python agent.py flappybird1 --train
```
8. run trained models
```
python agent.py flappybird1
```
