# Installation
(tested on Ubuntu 18.04 and 20.04)

1. install utility packages
2. clone the repo
3. set up a conda env
4. install python modules (including the main package for this repo, [mprl](../multiplayer-rl))

### Required packages
[tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) is used to launch and display parallel processes for paper experiments.  
[htop](https://htop.dev/) is a visual system process monitor.  
[git-lfs](https://git-lfs.github.com/) provides large file handling utilities for git
```shell script
# (On Ubuntu)
sudo apt update && sudo apt install tmux htop git-lfs
```

### Clone repo with git submodules
```shell script
git clone --recursive https://github.com/JBLanier/pipeline-psro.git
cd pipeline-psro
```
If you've already cloned this repo but not the [submodules](/dependencies), you can clone them with:
```shell script
git submodule update --init --recursive
```

This repo includes [neural network weights](/multiplayer-rl/mprl/data/learner_barrage_sac_arch1_pipeline_psro) tracked with git-lfs. If the initial clone wasn't rather large (several hundred megabytes), they may have not automatically been downloaded depending on your git version. Manually pull them just to be safe. 
```shell script
git lfs pull
```




### Set up Conda environment
After installing [Anaconda](https://docs.anaconda.com/anaconda/install/), enter the repo directory and create the new environment:
```shell script
conda env create -f environment.yml
conda activate p2sro_release
```

### Install Python modules

#### 1. DeepMind OpenSpiel (included dependency)
DeepMind's [Openspiel](https://github.com/deepmind/open_spiel) is used for Kuhn and Leduc Poker game logic as well as matrix-game utilities.

Perform the following steps with your conda env *active* to install OpenSpiel. (The conda env needs to be active so that OpenSpiel can find and compile against the python development headers in the env. Python version related issues may occur otherwise):

(Note, as of December 2022, the abseil-cpp `master` branch cloned by `./install.sh` will break compilation, so an additional command has been added below to clone a working fixed branch for abseil-cpp.)

```shell script
cd dependencies/open_spiel
./install.sh

# clone a working fixed branch for abseil-cpp
rm -rf open_spiel/abseil-cpp; git clone -b lts_2020_02_25 --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp

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
(The mprl package automatically adds these directories to the python system-path at runtime in its top-level [\_\_init.py__](../multiplayer-rl/mprl/__init__.py), so this step isn't necessary to run code in this repository.)

#### 2. Stratego Env Package (included dependency)
Stratego Multiplayer RL Environment
````shell script
cd dependencies/stratego_env
pip install -e .
cd ../..
````

#### 3. Multiplayer RL Package (main package)

```shell script
cd multiplayer-rl
pip install -e .
cd ..
```

Installation is now done!

### Check out Guides for Usage
1. [Running Poker Experiments](running_experiments.md)
2. [Playing against the Barrage Agent](barrage_agent.md)
