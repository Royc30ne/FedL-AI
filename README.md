# FedL-AI
A Open Source Federated Learning Simulation Framework for Everyone

![image](https://raw.githubusercontent.com/Royc30ne/FedL-AI/main/docs/imgs/FedLAI-logo.png)
## Introduction
This is a simulation framework for federated learning, based on TensorFlow. The framework allows for the simulation of various federated learning scenarios and the evaluation of different algorithms and models under those scenarios.

## What is Federated Learning?
Federated learning is a machine learning paradigm that allows for the training of models across multiple devices, without the need for centralized data storage. Instead, the models are trained locally on each device and only the model updates are sent to a central server. This approach has significant advantages in terms of data privacy and efficiency.

## How does the framework work?
The framework simulates a federated learning scenario by creating a set of virtual devices, each with its own local dataset. The devices communicate with a central server, exchanging model updates and receiving instructions on how to update their local models. The framework includes various algorithms and models for the devices to use, as well as metrics for evaluating their performance.

## Requirements
tensorflow >= 2.8.0
numpy == 1.24.2
scikit-learn == 1.2.1

## Reference
[multi-center-fed-learning](https://github.com/caifederated/multi-center-fed-learning)

[leaf](https://github.com/TalwalkarLab/leaf)


## Acknowledgements
Datasets used in this project is from [LEAF](https://leaf.cmu.edu/)

## Contributing
Contributions to the framework are welcome! If you would like to contribute, please fork the repository and submit a pull request.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.