# MLops Project

## Subject

In this repository, we decided to implement generative agents inspired by the paper [Generative Agents: Interactive Simulacra of Human Behavior](https://dl.acm.org/doi/pdf/10.1145/3586183.3606763).

Each agents must have a personality of its own, a memory of past events, opinions about the other agents, to be able to create plausible and interesting conversations, while being true to their character.

The code we used as a base, which we contributed to, can be found with detail explanations in the repository [NLPadvanced](https://github.com/Flooweur/NLPadvanced).

There is nevertheless an important difference with the base code, which is the way we interact with the LLM.  
Contrary to the previous version, we decided to use a model locally using Ollama, to be able to have more control on it, and to be able to use versioning tools for example.

## Features

The model can be deployed automatically within a production environnement, in this case a virtual machine created manually using VMware. It is easily accessible with ssh.

The deployement can be done using a simple command : `docker compose up --build`.

Among the proposed options to go further, we chose the following :

- we have an interface, enabling the user to use our project with ease
- we used canary deployement
- we used MLFlow to do model versioning and update the chosen model
- we implemented protection against prompt injection in our inputs
