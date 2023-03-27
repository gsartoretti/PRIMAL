# PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning

Reinforcement learning code to train multiple agents to
collaboratively plan their paths in a 2D grid world, as
well as to test/visualize the learned policy on handcrafted
scenarios.

**NEW**: Please try the [brand new online interactive
demo](https://primalgrid.netlify.app/primal) of our trained
PRIMAL model! You can customize the grid size, add/remove
obstacle, add agents and assign them goals, and finally
run the model online and see the results.

### File list

- DRLMAPF_A3C_RNN.ipynb: Multi-agent training code. Training
runs on GPU by default, change line "with tf.device("/gpu:0"):"
to "with tf.device("/cpu:0"):" to train on CPU (much slower).
- mapf_gym.py: Multi-agent path planning gym environment,
in which agents learn collective path planning.
- primal_testing.py: Code to run systematic validation tests
of PRIMAL, pulled from the saved_environments folder as .npy
files (examples available [here](https://drive.google.com/file/d/193mv6mhlcu9Bqxs6hSMTfSk_1GrPAiNO/view?usp=sharing)) and output results in a given
folder (by default: primal_results).
- mapf_gym_cap.py: Multi-agent path planning gym environment,
with capped goal distance state value for validation in
larger environments.
- mapgenerator.py: Script for creating custom environments and
testing a trained model on them. As an example, the trained
model used in our paper can be found [here](https://drive.google.com/file/d/1AtAeUwLF1Rn_X3b2FHkHi4fI5vveUHF6/view?usp=sharing).

## Before compilation: compile cpp_mstar code

- cd into the od_mstar3 folder.
- python3 setup.py build_ext (may need --inplace as extra argument).
- copy so object from build/lib.*/ at the root of the od_mstar3 folder.
- Check by going back to the root of the git folder,
running python3 and "import cpp_mstar"

### Custom testing

Edit mapgenerator.py to the correct path for the model.
By default, the model is loaded from the model_primal folder.

Hotkeys:
- o: obstacle mode
- a: agent mod
- g: goal mode, click an agent then click a free tile to place its goal
- c: clear agents
- r: reset
- up/down arrows: change size
- p: pause inference

### Requirements
- Python 3.4
- Cython 0.28.4
- OpenAI Gym 0.9.4
- Tensorflow 1.3.1
- Numpy 1.13.3
- matplotlib
- imageio (for GIFs creation)
- tk
- networkx (if using od_mstar.py and not the C++ version)

### Authors

[Guillaume Sartoretti](guillaume.sartoretti@gmail.com)

[Justin Kerr](jgkerr@andrew.cmu.edu)
