---
jupyter:
  colab:
    authorship_tag: ABX9TyN9xfXH8QSrAhZS4G2Nk8u0
    include_colab_link: true
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown colab_type="text" id="view-in-github"}
`<a href="https://colab.research.google.com/github/alazaradane/marl-robot-navigation/blob/main/Readme.ipynb" target="_parent">`{=html}`<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>`{=html}`</a>`{=html}
:::

::: {.cell .markdown id="WxCB3JeR1jBM"}
#ReadMe
:::

::: {.cell .markdown id="WH1eggRshYbj"}
# Drone Navigation with Reinforcement Learning (PPO)
:::

::: {.cell .markdown id="dqslUndShgJv"}
## intro
:::

::: {.cell .markdown id="oCmhtlq0kB-d"}
In this research project, we focus on developing a reinforcement
learning (RL)-based framework for autonomous drone navigation in a
dynamic 3D environment. By leveraging the power of PyBullet for
physics-based simulations and the Proximal Policy Optimization (PPO)
algorithm, we aim to design a system where drones can efficiently
navigate from a designated starting point to a target while avoiding
obstacles in their path.
:::

::: {.cell .markdown id="QifAUjSihkTa"}
This project implements a custom environment for drone navigation using
reinforcement learning (Proximal Policy Optimization - PPO). The
environment leverages PyBullet for realistic physics-based simulation,
allowing a drone to navigate from a dynamic starting point to a target
point while avoiding obstacles.
:::

::: {.cell .markdown id="y2KiDGwJhyvn"}
The main task for the reinforcement learning agent is as follows:

Goal: Navigate the drone from a randomly generated starting point to a
target point in a 3D space, while avoiding dynamically placed shaded
areas (obstacles). The shaded areas represent zones that must be
avoided, and they change in position for each episode. The drone is
rewarded for efficient navigation and penalized for: Colliding with
obstacles. Failing to reach the target within a predefined number of
steps. The action space involves continuous rotor forces to control the
drone\'s movement. Each episode challenges the agent to adjust its
navigation strategy based on the dynamic environment.
:::

::: {.cell .markdown id="jRPB2L-JkIe7"}
##Key Goals of the Research
:::

::: {.cell .markdown id="E3j28DbmkeTu"}
Dynamic Path Planning: Our primary goal is to teach drones to
autonomously compute and execute efficient navigation strategies in
real-time. The system incorporates dynamic obstacles and randomly
changing target locations, pushing the drone to adapt to environmental
changes while maintaining optimal performance.

Reinforcement Learning for Navigation: To achieve this, we employ
Proximal Policy Optimization (PPO), a state-of-the-art reinforcement
learning algorithm, which allows the drone to continuously improve its
performance by learning from interactions with the environment. Through
trial and error, the drone optimizes its rotor control, collision
avoidance, and trajectory planning.

Custom PyBullet Environment: The simulation environment is designed
using PyBullet, a versatile physics engine that provides real-time
simulations of rigid body dynamics. This setup allows for realistic
testing of drone behaviors, including rotor control, movement in a 3D
space, and interactions with obstacles, ensuring that the agent learns
to navigate effectively under realistic conditions.

Visualization and Analysis: The project includes visualizing the drone's
trajectory and monitoring its performance. This allows us to analyze the
efficiency and effectiveness of the trained RL agent in navigating
complex environments. By generating dynamic visual plots, we can track
the agent's learning progress and assess its ability to avoid obstacles
and reach targets in diverse scenarios.
:::

::: {.cell .markdown id="I67aqqRaicDj"}
##Overview
:::

::: {.cell .markdown id="PfEqAG_JijfA"}
The goal is to teach a drone how to navigate from a randomly generated
starting point to a target point while avoiding dynamically changing
obstacles. This project uses PyBullet for physics-based simulation and
TF-Agents for reinforcement learning.

By the end of this project, our target was to :

A fully functional custom environment for drone navigation. A trained
PPO agent capable of learning optimal navigation strategies.
Visualization of the drone\'s trajectory in a 3D space.
:::

::: {.cell .markdown id="1gDPIsYiiySM"}
##🧠 What's Our final actievements ?
:::

::: {.cell .markdown id="KSRQpAvtiyLM"}
Custom Environment Design:

Define observation and action spaces. Set up rewards and penalties for
drone navigation. Dynamically generate obstacles and targets. PPO
Algorithm:

Train an RL agent to control the drone\'s rotors. Use TF-Agents to
implement PPO. Simulation with PyBullet:

Visualize the drone, obstacles, and target in a 3D environment.
:::

::: {.cell .markdown id="-vJT81f7s-5S"}
Our focus on key challenges such as:

Efficient navigation with minimal communication. Handling dynamic and
unpredictable environments. Leveraging reinforcement learning to
autonomously improve drone performance over time.
:::

::: {.cell .markdown id="lwwXwsuatwVW"}
##Getting Started
:::

::: {.cell .markdown id="tUKQ4dMh2Y-i"}
###Dependencies & Packages:

------------------------------------------------------------------------
:::

::: {.cell .markdown id="WBIi9TKb2VHS"}
-   `<b>`{=html}`<a href="http://releases.ubuntu.com/16.04/">`{=html}Ubuntu
    16.04`</a>`{=html}`</b>`{=html}
-   `<b>`{=html}`<a href="http://wiki.ros.org/kinetic">`{=html}ROS
    Kinetic`</a>`{=html}`</b>`{=html}
-   `<b>`{=html}`<a href="http://gazebosim.org/">`{=html}Gazebo
    7`</a>`{=html}`</b>`{=html}
-   `<b>`{=html}`<a href="https://github.com/AutonomyLab/ardrone_autonomy">`{=html}ArDrone
    Autonomy ROS Package`</a>`{=html}`</b>`{=html}
-   `<b>`{=html}`<a href="https://gym.openai.com/docs/">`{=html}gym:
    0.9.3`</a>`{=html}`</b>`{=html}
-   `<b>`{=html}`<a href="https://www.tensorflow.org/install/">`{=html}TensorFLow
    1.1.0 (preferrable with GPU support)`</a>`{=html}`</b>`{=html}
:::

::: {.cell .markdown id="B5PnmSC0t5eg"}
###Prerequisites

------------------------------------------------------------------------
:::

::: {.cell .markdown id="BMdQBefQuA1H"}
Before you begin, ensure you have the following installed:

Python 3.8+

PyBullet (pip install pybullet)

TensorFlow (pip install tensorflow)

TF-Agents (pip install tf-agents)

NumPy, Matplotlib, and other required libraries (pip install numpy
matplotlib).
:::

::: {.cell .code id="YV1hJ29xu2IX"}
``` python
pip install pybullet
pip install tensorflow
pip install tf-agents
pip install numpy matplotlib

```
:::

::: {.cell .markdown id="fNZoFXqJzvyB"}
## Environmental Setup
:::

::: {.cell .markdown id="aeM_tkfz1Sx6"}
1.  Install Unreal Engine (4.27 suggested) from [Epic Games
    Launcher](https://store.epicgames.com/it/download).

2.  Install Visual Studio 2019

3.  Install C++ dev

4.  Install Python

5.  Download [AirSim](https://microsoft.github.io/AirSim/build_windows/)
    prebuilt source code and the environment of your choice.

6.  Place the Environment in AirSim/Unreal/Environment

7.  Use Visual Studio 2019 Developer Command Prompt with Admin
    privileges to run AirSim-1.7.0-windows/build.cmd

8.  Follow the
    [tutorial](https://microsoft.github.io/AirSim/unreal_blocks/) in
    order to setup Blocks Environment for AirSim

9.  Install [.net
    framework](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net462)
    4.6.2 Developer (SDK), desktop runtime 3.1.24

10. Run
    AirSim-1.7.0-windows/Unreal/Environments/Blocks/update_from_git.bat

11. Add settings.json inside airsim folder (settings.json is a file
    containing all the quadricopter settings)

12. Open .sln with Visual Studio 2022, as suggested in this
    [link](https://docs.microsoft.com/it-it/visualstudio/ide/how-to-set-multiple-startup-projects?view=vs-2022)
    set Blocks as default Project, DebugGame Editor & Win64. Finally
    press F5

13. Once Unreal is open with the project, click \"Play\" and use the
    keyboard to move the drone.
:::

::: {.cell .markdown id="iylMoRUF1fXb"}
###Python Interface with AirSim

1.  Take AirSim-1.7.0-windows/PythonClient/multirotor/hello_drone.py

2.  Delete first line of import.

3.  Create an Anaconda environment.

4.  Install the following libraries
    `bash     pip install numpy     pip install opencv-python     pip install msgpack-rpc-python     pip install airsim`

5.  Install Visual Studio & recommended python extensions (optional)

6.  Unreal might lag if there is another window on top.To avoid this go
    in Unreal Engine settings: Edit-\>Editor preferences-\>search
    Performance-\>disable \"Use less CPU when in background\"
:::

::: {.cell .markdown id="CaYD4fV_1rbN"}
## Run the project

1.  Clone the repository
    `bash     git clone https://github.com/lap98/RL-Drone-Stabilization.git`

2.  Open the environment in Unreal Engine

3.  Run first.py in order to control the drone
:::

::: {.cell .markdown id="gQOQ8s2d1-_P"}
## Reinforcement learning

In order to use TF-Agents library:

``` bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.9
pip install tf-agents==0.13.0
```
:::
