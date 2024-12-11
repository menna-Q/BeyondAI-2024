![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyN9xfXH8QSrAhZS4G2Nk8u0",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/alazaradane/marl-robot-navigation/blob/main/Readme.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#ReadMe"
      ],
      "metadata": {
        "id": "WxCB3JeR1jBM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Drone Navigation with Reinforcement Learning (PPO)"
      ],
      "metadata": {
        "id": "WH1eggRshYbj"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## intro"
      ],
      "metadata": {
        "id": "dqslUndShgJv"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "In this research project, we focus on developing a reinforcement learning (RL)-based framework for autonomous drone navigation in a dynamic 3D environment. By leveraging the power of PyBullet for physics-based simulations and the Proximal Policy Optimization (PPO) algorithm, we aim to design a system where drones can efficiently navigate from a designated starting point to a target while avoiding obstacles in their path."
      ],
      "metadata": {
        "id": "oCmhtlq0kB-d"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This project implements a custom environment for drone navigation using reinforcement learning (Proximal Policy Optimization - PPO). The environment leverages PyBullet for realistic physics-based simulation, allowing a drone to navigate from a dynamic starting point to a target point while avoiding obstacles.\n"
      ],
      "metadata": {
        "id": "QifAUjSihkTa"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "The main task for the reinforcement learning agent is as follows:\n",
        "\n",
        "Goal: Navigate the drone from a randomly generated starting point to a target point in a 3D space, while avoiding dynamically placed shaded areas (obstacles).\n",
        "The shaded areas represent zones that must be avoided, and they change in position for each episode.\n",
        "The drone is rewarded for efficient navigation and penalized for:\n",
        "Colliding with obstacles.\n",
        "Failing to reach the target within a predefined number of steps.\n",
        "The action space involves continuous rotor forces to control the drone's movement.\n",
        "Each episode challenges the agent to adjust its navigation strategy based on the dynamic environment.\n",
        "\n"
      ],
      "metadata": {
        "id": "y2KiDGwJhyvn"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Key Goals of the Research"
      ],
      "metadata": {
        "id": "jRPB2L-JkIe7"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Dynamic Path Planning:\n",
        "Our primary goal is to teach drones to autonomously compute and execute efficient navigation strategies in real-time. The system incorporates dynamic obstacles and randomly changing target locations, pushing the drone to adapt to environmental changes while maintaining optimal performance.\n",
        "\n",
        "Reinforcement Learning for Navigation:\n",
        "To achieve this, we employ Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm, which allows the drone to continuously improve its performance by learning from interactions with the environment. Through trial and error, the drone optimizes its rotor control, collision avoidance, and trajectory planning.\n",
        "\n",
        "Custom PyBullet Environment:\n",
        "The simulation environment is designed using PyBullet, a versatile physics engine that provides real-time simulations of rigid body dynamics. This setup allows for realistic testing of drone behaviors, including rotor control, movement in a 3D space, and interactions with obstacles, ensuring that the agent learns to navigate effectively under realistic conditions.\n",
        "\n",
        "Visualization and Analysis:\n",
        "The project includes visualizing the drone’s trajectory and monitoring its performance. This allows us to analyze the efficiency and effectiveness of the trained RL agent in navigating complex environments. By generating dynamic visual plots, we can track the agent’s learning progress and assess its ability to avoid obstacles and reach targets in diverse scenarios."
      ],
      "metadata": {
        "id": "E3j28DbmkeTu"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Overview"
      ],
      "metadata": {
        "id": "I67aqqRaicDj"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "The goal is to teach a drone how to navigate from a randomly generated starting point to a target point while avoiding dynamically changing obstacles. This project uses PyBullet for physics-based simulation and TF-Agents for reinforcement learning.\n",
        "\n",
        "By the end of this project, our target was to :\n",
        "\n",
        "A fully functional custom environment for drone navigation.\n",
        "A trained PPO agent capable of learning optimal navigation strategies.\n",
        "Visualization of the drone's trajectory in a 3D space.\n"
      ],
      "metadata": {
        "id": "PfEqAG_JijfA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##🧠 What’s Our final actievements ?"
      ],
      "metadata": {
        "id": "1gDPIsYiiySM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Custom Environment Design:\n",
        "\n",
        "Define observation and action spaces.\n",
        "Set up rewards and penalties for drone navigation.\n",
        "Dynamically generate obstacles and targets.\n",
        "PPO Algorithm:\n",
        "\n",
        "Train an RL agent to control the drone's rotors.\n",
        "Use TF-Agents to implement PPO.\n",
        "Simulation with PyBullet:\n",
        "\n",
        "Visualize the drone, obstacles, and target in a 3D environment.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "KSRQpAvtiyLM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        " Our focus on key challenges such as:\n",
        "\n",
        "Efficient navigation with minimal communication.\n",
        "Handling dynamic and unpredictable environments.\n",
        "Leveraging reinforcement learning to autonomously improve drone performance over time."
      ],
      "metadata": {
        "id": "-vJT81f7s-5S"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Getting Started"
      ],
      "metadata": {
        "id": "lwwXwsuatwVW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Dependencies & Packages:\n",
        "\n",
        "---\n"
      ],
      "metadata": {
        "id": "tUKQ4dMh2Y-i"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "- <b><a href=\"http://releases.ubuntu.com/16.04/\">Ubuntu 16.04</a></b>\n",
        "- <b><a href=\"http://wiki.ros.org/kinetic\">ROS Kinetic</a></b>\n",
        "- <b><a href=\"http://gazebosim.org/\">Gazebo 7</a></b>\n",
        "- <b><a href=\"https://github.com/AutonomyLab/ardrone_autonomy\">ArDrone Autonomy ROS Package</a></b>\n",
        "- <b><a href=\"https://gym.openai.com/docs/\">gym: 0.9.3</a></b>\n",
        "- <b><a href=\"https://www.tensorflow.org/install/\">TensorFLow 1.1.0 (preferrable with GPU support)</a></b>\n"
      ],
      "metadata": {
        "id": "WBIi9TKb2VHS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Prerequisites\n",
        "\n",
        "\n",
        "---\n"
      ],
      "metadata": {
        "id": "B5PnmSC0t5eg"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Before you begin, ensure you have the following installed:\n",
        "\n",
        "Python 3.8+\n",
        "\n",
        "PyBullet (pip install pybullet)\n",
        "\n",
        "TensorFlow (pip install tensorflow)\n",
        "\n",
        "TF-Agents (pip install tf-agents)\n",
        "\n",
        "NumPy, Matplotlib, and other required libraries (pip install numpy matplotlib)."
      ],
      "metadata": {
        "id": "BMdQBefQuA1H"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install pybullet\n",
        "pip install tensorflow\n",
        "pip install tf-agents\n",
        "pip install numpy matplotlib\n",
        "\n"
      ],
      "metadata": {
        "id": "YV1hJ29xu2IX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "## Environmental Setup\n"
      ],
      "metadata": {
        "id": "fNZoFXqJzvyB"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "1. Install Unreal Engine (4.27 suggested) from [Epic Games Launcher](https://store.epicgames.com/it/download).\n",
        "\n",
        "2. Install Visual Studio 2019\n",
        "\n",
        "3. Install C++ dev\n",
        "\n",
        "4. Install Python\n",
        "\n",
        "5. Download [AirSim](https://microsoft.github.io/AirSim/build_windows/) prebuilt source code and the environment of your choice.\n",
        "\n",
        "6. Place the Environment in AirSim/Unreal/Environment\n",
        "\n",
        "5. Use Visual Studio 2019 Developer Command Prompt with Admin privileges to run AirSim-1.7.0-windows/build.cmd\n",
        "\n",
        "6. Follow the [tutorial](https://microsoft.github.io/AirSim/unreal_blocks/) in order to setup Blocks Environment for AirSim\n",
        "\n",
        "7. Install [.net framework](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net462) 4.6.2 Developer (SDK), desktop runtime 3.1.24\n",
        "\n",
        "8. Run AirSim-1.7.0-windows/Unreal/Environments/Blocks/update_from_git.bat\n",
        "\n",
        "9. Add settings.json inside airsim folder (settings.json is a file containing all the quadricopter settings)\n",
        "\n",
        "10. Open .sln with Visual Studio 2022, as suggested in this [link](https://docs.microsoft.com/it-it/visualstudio/ide/how-to-set-multiple-startup-projects?view=vs-2022) set Blocks as default Project, DebugGame Editor & Win64. Finally press F5\n",
        "\n",
        "11. Once Unreal is open with the project, click \"Play\" and use the keyboard to move the drone.\n",
        "\n"
      ],
      "metadata": {
        "id": "aeM_tkfz1Sx6"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Python Interface with AirSim\n",
        "\n",
        "1. Take AirSim-1.7.0-windows/PythonClient/multirotor/hello_drone.py\n",
        "\n",
        "2. Delete first line of import.\n",
        "\n",
        "3. Create an Anaconda environment.\n",
        "\n",
        "4. Install the following libraries\n",
        "    ```bash\n",
        "    pip install numpy\n",
        "    pip install opencv-python\n",
        "    pip install msgpack-rpc-python\n",
        "    pip install airsim\n",
        "    ```\n",
        "5. Install Visual Studio & recommended python extensions (optional)\n",
        "\n",
        "6. Unreal might lag if there is another window on top.To avoid this go in Unreal Engine settings: Edit->Editor preferences->search Performance->disable \"Use less CPU when in background\"\n",
        "\n"
      ],
      "metadata": {
        "id": "iylMoRUF1fXb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Run the project\n",
        "1. Clone the repository\n",
        "    ```bash\n",
        "    git clone https://github.com/lap98/RL-Drone-Stabilization.git\n",
        "    ```\n",
        "2. Open the environment in Unreal Engine\n",
        "\n",
        "3. Run first.py in order to control the drone\n",
        "\n"
      ],
      "metadata": {
        "id": "CaYD4fV_1rbN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Reinforcement learning\n",
        "\n",
        "In order to use TF-Agents library:\n",
        "```bash\n",
        "conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0\n",
        "pip install tensorflow==2.9\n",
        "pip install tf-agents==0.13.0\n",
        "```"
      ],
      "metadata": {
        "id": "gQOQ8s2d1-_P"
      }
    }
  ]
}
