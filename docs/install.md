# Installation
(tested on Ubuntu 18.04 and 20.04)

### Required packages
[tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) is used to launch and display parallel processes for paper experiments:
```shell script
# (On Ubuntu)
sudo apt update && sudo apt install tmux
```

### Clone repo with git submodules
```shell script
git clone --recursive https://github.com/JBLanier/distributed-rl-for-imperfect-info-games.git
```
If you've already cloned this repo but not the submodules (located in the dependencies directory), you can clone them with:
```shell script
git submodule update --init --recursive
```


### Set up Conda environment
After installing [Anaconda](https://docs.anaconda.com/anaconda/install/):
```shell script
conda env create -f environment.yml
conda activate p2sro_release
```

### Install Python modules

#### 1. DeepMind OpenSpiel (included dependency)
DeepMind's [Openspiel](https://github.com/deepmind/open_spiel) is used for Kuhn and Leduc Poker game logic as well as matrix-game utilities.

With your conda env active:
```shell script
cd dependencies/open_spiel
./install.sh
mkdir build
cd build
CXX=g++ cmake -DPython_TARGET_VERSION=3.6 -DCMAKE_CXX_COMPILER=${CXX} -DPython3_FIND_VIRTUALENV=FIRST -DPython3_FIND_STRATEGY=LOCATION ../open_spiel
make -j$(nproc)
cd ../../..
```

(Optional) To import OpenSpiel for your own use independently of the mprl package, add OpenSpiel directories to your PYTHONPATH in your ~/.bashrc ([more details here](https://github.com/deepmind/open_spiel/blob/222ba03f73d643658838d0d95331e9c8a4f77cf1/docs/install.md)):
```shell script
# Add the following lines to your ~/.bashrc:
# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>/build/python
```
(The mprl package automatically adds these directories to the python system-path in its top-level \_\_init.py__, so this step isn't necessary to run code in this repository.)

#### 2. Stratego Env (included dependency)
Stratego Multiplayer RL Environment
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


### Checkout Guides for Usage
1. [Running Poker Experiments](running_experiments.md)
2. [Playing against the Barrage Agent](barrage_agent.md)
