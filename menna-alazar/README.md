![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

# Multi-Agent Reinforcement Learning for Robot Navigation

Hi there! 👋 Welcome to our research project on Multi-Agent Reinforcement Learning for Robot Navigation. In this research, we focus on Single-Agent Q-learning applied to different environments and multi agents communications techniques. By the end of this , we'll analyze the results and explore how our agent navigates through the environments.

<a href="https://colab.research.google.com/github/alazaradane/marl-robot-navigation/blob/main/Readme.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



### Introdution

In this research project, we focus on developing a reinforcement learning (RL)-based framework for autonomous drone navigation in a dynamic 3D environment. By leveraging the power of PyBullet for physics-based simulations and the Proximal Policy Optimization (PPO) algorithm, we aim to design a system where drones can efficiently navigate from a designated starting point to a target while avoiding obstacles in their path.

This project implements a custom environment for drone navigation using reinforcement learning (Proximal Policy Optimization - PPO). The environment leverages PyBullet for realistic physics-based simulation, allowing a drone to navigate from a dynamic starting point to a target point while avoiding obstacles.

The main task for the reinforcement learning agent is as follows:

- **Goal**: Navigate the drone from a randomly generated starting point to a target point in a 3D space, while avoiding dynamically placed shaded areas (obstacles).
- The shaded areas represent zones that must be avoided, and they change in position for each episode.
- The drone is rewarded for efficient navigation and penalized for:
  - Colliding with obstacles.
  - Failing to reach the target within a predefined number of steps.
- The action space involves continuous rotor forces to control the drone's movement.
- Each episode challenges the agent to adjust its navigation strategy based on the dynamic environment.

### Key Goals of the Research

1. **Dynamic Path Planning**:
   - Teach drones to autonomously compute and execute efficient navigation strategies in real-time.
   - Incorporate dynamic obstacles and randomly changing target locations, pushing the drone to adapt to environmental changes while maintaining optimal performance.

2. **Reinforcement Learning for Navigation**:
   - Employ Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm, to continuously improve drone performance by learning from interactions with the environment.
   - Optimize rotor control, collision avoidance, and trajectory planning through trial and error.

3. **Custom PyBullet Environment**:
   - Design a simulation environment using PyBullet for realistic testing of drone behaviors, including rotor control, movement in a 3D space, and interactions with obstacles.

4. **Visualization and Analysis**:
   - Visualize the drone’s trajectory and monitor its performance.
   - Analyze the efficiency and effectiveness of the trained RL agent in navigating complex environments.
   - Generate dynamic visual plots to track the agent’s learning progress.

### Overview

The goal is to teach a drone how to navigate from a randomly generated starting point to a target point while avoiding dynamically changing obstacles. This project uses PyBullet for physics-based simulation and TF-Agents for reinforcement learning.

By the end of this project, our target was to:

- Build a fully functional custom environment for drone navigation.
- Train a PPO agent capable of learning optimal navigation strategies.
- Visualize the drone's trajectory in a 3D space.

### 🧪 What’s Our Final Achievements?

- **Custom Environment Design**:
  - Define observation and action spaces.
  - Set up rewards and penalties for drone navigation.
  - Dynamically generate obstacles and targets.

- **PPO Algorithm**:
  - Train an RL agent to control the drone's rotors.
  - Use TF-Agents to implement PPO.

- **Simulation with PyBullet**:
  - Visualize the drone, obstacles, and target in a 3D environment.

### Challenges

- Efficient navigation with minimal communication.
- Handling dynamic and unpredictable environments.
- Leveraging reinforcement learning to autonomously improve drone performance over time.

## Getting Started

### Dependencies & Packages:

- [Ubuntu 16.04](http://releases.ubuntu.com/16.04/)
- [ROS Kinetic](http://wiki.ros.org/kinetic)
- [Gazebo 7](http://gazebosim.org/)
- [ArDrone Autonomy ROS Package](https://github.com/AutonomyLab/ardrone_autonomy)
- [gym: 0.9.3](https://gym.openai.com/docs/)
- [TensorFlow 1.1.0](https://www.tensorflow.org/install/) (preferably with GPU support)

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8+
- PyBullet: `pip install pybullet`
- TensorFlow: `pip install tensorflow`
- TF-Agents: `pip install tf-agents`
- NumPy and Matplotlib: `pip install numpy matplotlib`

### Installation Commands

```bash
pip install pybullet
pip install tensorflow
pip install tf-agents
pip install numpy matplotlib
```

## Environmental Setup

1. Install Unreal Engine (4.27 suggested) from [Epic Games Launcher](https://store.epicgames.com/it/download).
2. Install Visual Studio 2019.
3. Install C++ development tools.
4. Install Python.
5. Download [AirSim](https://microsoft.github.io/AirSim/build_windows/) prebuilt source code and the environment of your choice.
6. Place the Environment in `AirSim/Unreal/Environment`.
7. Use Visual Studio 2019 Developer Command Prompt with Admin privileges to run `AirSim-1.7.0-windows/build.cmd`.
8. Follow the [tutorial](https://microsoft.github.io/AirSim/unreal_blocks/) to set up the Blocks Environment for AirSim.
9. Install [.NET Framework](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net462) 4.6.2 Developer (SDK), desktop runtime 3.1.24.
10. Run `AirSim-1.7.0-windows/Unreal/Environments/Blocks/update_from_git.bat`.
11. Add a `settings.json` file inside the AirSim folder (containing quadcopter settings).
12. Open `.sln` file with Visual Studio 2022, set Blocks as the default Project, DebugGame Editor & Win64. Finally, press F5.
13. Once Unreal is open with the project, click "Play" and use the keyboard to move the drone.

### Python Interface with AirSim

1. Take `AirSim-1.7.0-windows/PythonClient/multirotor/hello_drone.py`.
2. Delete the first line of the import.
3. Create an Anaconda environment.
4. Install the following libraries:

```bash
pip install numpy
pip install opencv-python
pip install msgpack-rpc-python
pip install airsim
```

5. Install Visual Studio & recommended Python extensions (optional).
6. If Unreal lags when in the background, go to Unreal Engine settings: Edit -> Editor Preferences -> Search "Performance" -> Disable "Use less CPU when in background".

## Run the Project

1. Clone the repository:

```bash
git clone https://github.com/lap98/RL-Drone-Stabilization.git
```

2. Open the environment in Unreal Engine.
3. Run `first.py` to control the drone.

## Reinforcement Learning

To use the TF-Agents library, install:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.9
pip install tf-agents==0.13.0 
```

## Conclusion

This project demonstrates the successful implementation of a reinforcement learning-based framework for autonomous drone navigation. By using the Proximal Policy Optimization algorithm and a physics-based simulation environment powered by PyBullet, the drone was trained to navigate efficiently in dynamic and obstacle-filled 3D environments.


> The research poster for this project can be found in the [BeyondAI Proceedings 2024](https://thinkingbeyond.education/beyondai_proceedings_2024/).




