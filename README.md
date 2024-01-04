# Semantic Search-based online policy
[Project page](https://j-sheikh.github.io/behavioral-search-policy/) | [Video](https://www.youtube.com/watch?v=IEEhlYKjs-E&feature=youtu.be) | [arXiv](https://arxiv.org/abs/2312.05925)

Authors: Jannik Sheikh, Andrew Melnik, Gora Chand Nandi, Robert Haschke

<p align="center">
	<img src="assets/splash.gif" />
</p>

Abstract:

Reinforcement learning and Imitation Learning approaches utilize policy learning strategies that are difficult to generalize well with just a few examples of a task. In this work, we propose a language-conditioned semantic search-based method to produce an online search-based policy from the available demonstration dataset of state-action trajectories. Here we directly acquire actions from the most similar manipulation trajectories found in the dataset. Our approach surpasses the performance of the baselines on the CALVIN benchmark and exhibits strong zero-shot adaptation capabilities. This holds great potential for expanding the use of our online search-based policy approach to tasks typically addressed by Imitation Learning or Reinforcement Learning-based policies.


## Installation


### 1. Clone Repository with Submodules

```bash
# Clone the repository with the submodule
git clone --recursive https://github.com/j-sheikh/behavioral-search-policy
```

### 2. Install 'calvin' and 'hulc'

#### For more details check the corresponding repos [CALVIN](https://github.com/mees/calvin) and [HULC](https://github.com/lukashermann/hulc). 
```bash
# Navigate to the 'calvin' submodule and run installation script
cd calvin
sh install.sh
cd ..

# Navigate to the 'hulc' submodule and run installation script
cd hulc
sh install.sh
```
## Evaluation

Follow the instructions in the [HULC repo](https://github.com/lukashermann/hulc). For evaluating our search-based policy reference in hulc/evaluation/evaluate_policy.py to either sb_model/evaluation/evaluation_netural_position.py or sb_model/evaluation/evaluation_single.py. 

