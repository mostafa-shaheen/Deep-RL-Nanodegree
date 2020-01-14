# Tennis player
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/0689fca92f594fe88299d1e95d7e4c48)](https://www.codacy.com/manual/fg/drlnd-p3-collaboration-competition?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/drlnd-p3-collaboration-competition&amp;utm_campaign=Badge_Grade) [![CircleCI](https://circleci.com/gh/frgfm/drlnd-p3-collaboration-competition.svg?style=shield)](https://circleci.com/gh/frgfm/drlnd-p3-collaboration-competition)

This repository is an implementation of DDPG agents for the collaboration and competition project of Udacity Deep Reinforcement Learning nanodegree, in the tennis game provided by unity.

![tennis](https://github.com/mostafa-shaheen/Deep-RL-Nanodegree/blob/master/p3_collaboration-competition/Tennis_player.gif)



## Table of Contents

- [Environment](#environment)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [instructions](#instructions)


## Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)
- [ml-agents](https://github.com/Unity-Technologies/ml-agents) v0.4 (check this [release](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0b) if you encounter issues)

### Installation

You can install the project requirements as follows:

```shell
git clone https://github.com/frgfm/drlnd-p3-collaboration-competition.git
cd drlnd-p3-collaboration-competition
pip install -r requirements.txt
```

Download the environment build corresponding to your OS

- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then extract the archive in the project folder.


Place the file in the DRLND GitHub repository, in the p3_collaboration-competition/ folder, and unzip (or decompress) the file.

## Instructions
Follow the instructions in Tennis.ipynb to get started with training your own agent!
