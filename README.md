# Open-Source AI Models: Opportunities and Challenges of Collaborative Software Learning

## Framework
We use [Flower](https://flower.ai/) as our Federated Framework. You can change the NUM_CLIENTS in client.py

## datasets
We use the datasets from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). We choose five tasks from CodeXGLUE as our experiments: Clone-detection, Defect-Detection, Code-to-Text, NL-code-search-Adv, CodeCompletion-token, which provide a comprehensive overview of how models perform in real-world programming scenarios, highlighting their abilities in understanding, generating and manipulating code.

Please split the datasets into different part before using Federated Learning.

## models
We use four models in our experiment: [CodeBERT](https://huggingface.co/microsoft/codebert-base), [CodeT5](https://huggingface.co/Salesforce/codet5-base), [CodeGPT](https://huggingface.co/microsoft/CodeGPT-small-java), [CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)

![model](https://github.com/microsoft/CodeXGLUE/raw/main/baselines.jpg)

## quickstart
```bash
conda create --name FL python=3.10
pip install -r requirements.txt
```

## Other task
We provide a framework you can use directly to train your model with federated learning.