# LLM-Finetuner
this repo is dedicated for a framework that facilitates the process of fine tuning LLMs from data preprocessing to evaluation with an emphasize on efficiency and cost-effectiveness



## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This framework consists of the following modules:
- Preprocessing: where you provide a link to a huggingface hosted dataset and define the preprocessing criterias
- Fine-tuning: where you specify the base LLM and the processed dataset as well as the training hyperparameters
- Generation: where you can test your fine-tuned model and generate answers and make conversations
- Evaluation: many evaluation metrics are available where you can evaluate your finetuned model on many aspects
  

## Features

- Easy setup and configuration
- Support for huggingface models and datasets
- Comprehensive experiment management
- Detailed performance evaluation and reporting through **weights and biases** integration
- Customizable and extensible

## Installation

To install LLM-Finetuner, clone the repository and install the required dependencies:

```bash
git clone https://github.com/imadken/LLM-Finetuner.git
cd LLM-Finetuner

```

## Usage

Follow these steps to use LLM-Finetuner for your fine-tuning tasks:

### 1. Clone the Project

### 2. setup conda env

create a conda env from ```environment.yml```

```bash
conda env create -f environment.yml

conda activate myenv

conda env list
```
conda activate myenv


if you need to evaluate using LLM-based evaluation then you should switch to  ```environment-test.yml``` 

### 3. Configure the necessary Yaml files

- For each module, there is a yaml file to configure
- Before the evaluation, you need to generate answers and feed them to the evaluation module
- in the end, configure the ```main.yaml``` and run ```main.py```

## Contributing

We welcome contributions to LLM-Finetuner! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more detailed documentation and examples, please refer to the project's for any questions or support.
