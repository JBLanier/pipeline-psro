Code Release for [Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games](https://arxiv.org/abs/2006.08555)

# Installation
(tested on Ubuntu 18.04 and 20.04)

### Required packages
[tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) is used to launch and display paper experiments:
```shell script
# (On Ubuntu)
sudo apt update && sudo apt install tmux
```

### Clone repo with git submodules
```shell script
git clone --recursive https://github.com/JBLanier/distributed-rl-for-imperfect-info-games.git
```
If you've already cloned this repo but not the submodules located in the dependencies directory, you can clone them with:
```shell script
git submodule update --init --recursive
```


### Set up Conda environment
```shell script
conda env create -f environment.yml
conda activate p2sro
```

### Install Python modules

#### 1. DeepMind OpenSpiel (included dependency)
DeepMind's [Openspiel](https://github.com/deepmind/open_spiel) is used for Kuhn and Leduc Poker game logic as well as matrix-game utilities.
```shell script
cd depedencies/open_spiel
./install.sh
cd ../..
```
(If you're familiar with OpenSpiel, you may be wondering if you need to manually include it in your PYTHONPATH. The mprl package automatically includes it in its top-level \_\_init.py__ )

#### 2. Stratego Env (included dependency)
Stratego Multiplayer RL Evironment
````shell script
cd dependencies/stratego_env
pip install -e .
cd ../..
````

#### 3. Multiplayer RL (main package)

```shell script
cd multiplayer-rl
pip install -e .
```

# Running experiments