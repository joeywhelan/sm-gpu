# Elastic/Nvidia GPU Integration
## Contents
1.  [Summary](#summary)
2.  [Architecture](#architecture)
3.  [Features](#features)
4.  [Prerequisites](#prerequisites)
5.  [Installation](#installation)
6.  [Usage](#usage)

## Summary <a name="summary"></a>
This is a demonstration of integration of Nvidia GPUs to self-managed Elastic Cloud on Kubernetes (ECK) for the purpose of acceleration of embeddings and indexing.

## Architecture <a name="architecture"></a>
![architecture](assets/highlevel.png) 


## Features <a name="features"></a>
- Jupyter notebook
- Builds an ECK deployment on Google Kubernetes Engine (GKE)
- GKE deployment includes CPU-only nodes for the Elastic Master nodes and CPU + Nvidia GPU nodes for the Elastic Data nodes.
- Creates a synthetic dataset with a text field and dense vector field from jina-embeddings-v3
- Conducts a CPU vs GPU test of indexing on that dataset
- Deletes the entire GKE environment

## Prerequisites <a name="prerequisites"></a>
- GCP project
- gcloud CLI
- Jina API key
- Python

## Installation <a name="installation"></a>
- Edit the .api_keys.template file and rename to .api_keys
- Create a Python virtual environment

## Usage <a name="usage"></a>
- Execute notebook