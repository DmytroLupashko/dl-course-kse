# KSE Deep Learning Course

# Setup Working Environment  

## Pre-requirements 

- [Poetry](https://python-poetry.org/docs/#installation).
- (Optional) [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). I advice using Miniconda
- VS Code — [Ubuntu](https://code.visualstudio.com/docs/setup/linux), [macOS](https://code.visualstudio.com/docs/setup/mac) and [Windows](https://code.visualstudio.com/docs/setup/windows) installation guides.
- (Optional) CUDA Version: 11.4; Driver Version: 470.129.06 — [Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## Setup environment 

### Poetry (Main)

1. Install Poetry using [Poetry full guide](https://python-poetry.org/docs/#installation).
    - The easiest way is to use the [Official Installer guide](https://python-poetry.org/docs/#installing-with-the-official-installer).
    - Pay attention to `poetry --version`. To install the correct version, run: `curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3`.
    - If you have already installed another version, simply change it with: `poetry self update 2.1.3`.
2. Configure Poetry to create Env in local folder - `poetry config virtualenvs.in-project true`.
3. Activate the environment: `eval $(poetry env activate)`.
    - If you have `conda` and 2 environments were activated: `conda deactivate`.
4. `poetry install --no-root`.
6. In order to deactivate env - `exit`.
7. In order to remove env - `rm .venv -rf`.

In order to activate environment on the next use:

`eval $(poetry env activate)`

> **Important**: you should be inside your project folder to do it.

# Start Jupyter

```bash
jupyter lab --port 7766
```

> **Note**: you may use any port.

# Content 

1. [] Deep Learning Basics
    1. [x] Deep Learning Overview, PyTorch, Perceptron
        - Author: Volodymyr
    2. [x] Backpropagation & Advanced Optimization
        - Author: Volodymyr
    3. [x] Validation & Metrics
        - Author: Anton
    4. [] Regularization: Dropout, L1/L2, Batch/Layer Normalization
        - Author: Volodymyr
    5. [] Frameworks & Logging: PyTorch Lightning & Weights & Biases
        - Author: Anton
    6. [x] Homework
2. [] Convolutional Neural Networks (CNNs)
    1. [] Intro to Image Processing, Convolutional & Pooling Layers 
        - Author: Volodymyr
    2. [] ResNets & EfficientNets
        - Author: Volodymyr
    3. [] CNNs applications for Audio ML
        - Author: Volodymyr
    4. Homework
3. []  Recurrent Neural Networks (RNNs)
    1. [] Intro to Sequence Tasks, Representations & Embeddings
        - Author: Anton
    2. [] RNNs 
        - Author: Anton
    3. [] GRUs & LSTMs 
        - Author: Anton
    4. Homework
4. [] Transformers
    1. [] Attention from Scratch, Encoder Transformer.
        - Author: Anton
    2. [] Encoder–Decoder Transformer
        - Author: Anton
    3. [] Transformers Beyond Text
        - Author: Anton
    4. [] Kaggle “Black Magic” Fine-Tuning & Distillation
        - Author: Volodymyr
    5. Homework

# Use Kaggle or Colab for computations

## Kaggle 

1. Create a [Kaggle](https://www.kaggle.com/) account.
2. Create a [Notebook](https://www.kaggle.com/code).
3. Explore the [docs](https://www.kaggle.com/docs/notebooks) and find out how to:
    - Add the Kaggle dataset to the notebook.
    - Turn on GPU.

## Colab 

1. Create a Notebook in [Colab](https://colab.research.google.com/).
2. Enable GPU.
3. Add the Kaggle dataset to Colab following the [guide](https://www.geeksforgeeks.org/how-to-import-kaggle-datasets-directly-into-google-colab/).

# Data

## How to use Kaggle datasets

1. Create a [Kaggle](https://www.kaggle.com/) account.
2. Proceed [with Installation & Authentication](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).
3. Don't forget to join a competition and accept its rules on a Kaggle website.
4. Download the dataset with an API command.

# Citation

```
@misc{kse_deep_learning,
  author = {Volodymyr Sydorskyi, Anton Bazdyrev},
  title = {UCU Deep Learning Course},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI301-Deep-Learning/kse_deep_learning}},
}
```
