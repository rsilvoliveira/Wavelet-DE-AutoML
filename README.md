# Wavelet-DE-AutoML

A novel automated framework for accurate streamflow forecasting that integrates wavelet transforms, Differential Evolution optimization, and Automated Machine Learning.

## Description

[Project description to be added here.]

## How It Works

This framework is designed to automate the process of building high-performance time series forecasting models. It takes a raw time series as input and uses a Differential Evolution (DE) metaheuristic to explore a vast search space of possible modeling pipelines.

The core functionality includes:

* **Automated Preprocessing**: Optimally selects whether to use a Wavelet Transform, and if so, which wavelet filter (`db1`, `bior1.1`, etc.) and decomposition level to apply.
* **Automated Feature Engineering**: Determines the optimal number of past time steps (look-back window) to use as input features for the model.
* **Automated Model Selection**: Chooses the best regression model from a predefined pool of candidates (e.g., MLP, XGBoost, LSSVR, SVR, KNN, Elastic Net).
* **Automated Hyperparameter Tuning**: Simultaneously tunes the hyperparameters of the selected regression model.

The goal is to find the single best combination of preprocessing, features, model, and hyperparameters that minimizes forecasting error for a given time series and forecast horizon.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

The following prerequisites are required to run the project:

* **Python 3.10**
* All other library dependencies are listed in the `requirements.txt` file.

> [!NOTE]
> It is highly recommended to create a dedicated virtual environment (e.g., Conda or venv) before installing the dependencies to avoid conflicts with other projects.

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/rsilvoliveira/Wavelet-DE-AutoML.git
    ```

2.  Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```


### Using Auto-sklearn and AutoKeras

This repository also includes separate implementations using **Auto-sklearn** and **AutoKeras**. To run these examples, you must install these libraries manually by following their official guides.

* [Auto-sklearn Installation Guide](https://automl.github.io/auto-sklearn/stable/installation.html)
* [AutoKeras Installation Guide](https://autokeras.com/install/)

> [!NOTE]
> These libraries have specific dependencies (e.g., Python versions, build tools, ML frameworks) that may differ from the core requirements of this project. Please review their documentation carefully before proceeding with the installation.


## How to Run

[Execution instructions to be added here.]

## Project Organization

The repository is structured to separate data, configuration, and the core logic of the optimization and modeling framework.

<details open>
<summary><strong>Complete Project Structure (Click to expand/collapse)</strong></summary>
<ul>
    <li>
         <strong>autoKeras</strong> - Contains AutoKeras implementations for automated ML.     
    </li>
    <li>        
        <strong>autoSklearn</strong> - Contains Auto-sklearn implementations for automated ML.
    </li>
    <li>
        <details>
            <summary><strong>database/</strong> - Contains the time series datasets.</summary>
            <ul>
                <li>
                    <details>
                        <summary><strong>basins/</strong> - Data from river basins.</summary>
                        <ul>
                            <li><strong>daily/</strong></li>
                            <li><strong>hourly/</strong></li>
                            <li><strong>monthly/</strong></li>
                        </ul>
                    </details>
                </li>
                <li>
                    <details>
                        <summary><strong>hydroelectric plants/</strong> - Data from hydroelectric plants.</summary>
                        <ul>
                            <li><strong>daily/</strong></li>
                            <li><strong>monthly/</strong></li>
                        </ul>
                    </details>
                </li>
                <li>
                    <details>
                        <summary><strong>stations/</strong> - Data from monitoring stations.</summary>
                        <ul>
                            <li><strong>daily/</strong></li>
                            <li><strong>monthly/</strong></li>
                        </ul>
                    </details>
                </li>
            </ul>
        </details>
    </li>
    <li>
        <details>
            <summary><strong>util/</strong> - Contains utility modules for heuristics and regressors.</summary>
            <ul>
                <li>
                    <details>
                        <summary><strong>heuristics/</strong> - Implementations of metaheuristic algorithms.</summary>
                        <ul>
                            <li><code>chicken_so.py</code></li>
                            <li><code>de.py</code></li>
                            <li><code>fda.py</code></li>
                            <li><code>ga.py</code></li>
                            <li><code>gwo.py</code></li>
                            <li><code>i_gwo.py</code></li>
                            <li><code>__init__.py</code></li>
                            <li><code>pso.py</code></li>
                            <li><code>sa.py</code></li>
                        </ul>
                    </details>
                </li>
                <li>
                    <details>
                        <summary><strong>regressors/</strong> - Implementation of the machine learning models.</summary>
                        <ul>
                            <li><code>auto_ml.py</code></li>
                            <li><code>conv_lstm_regressor.py</code></li>
                            <li><code>ELM.py</code></li>
                            <li><code>elm_regressor.py</code></li>
                            <li><code>elm_regressor_ver2.py</code></li>
                            <li><code>__init__.py</code></li>
                            <li><code>knn_regressor.py</code></li>
                            <li><strong>linear/</strong></li>
                            <li><code>lssvr_regressor.py</code></li>
                            <li><code>lssvr_regressor_ver2.py</code></li>
                            <li><code>lstm_regressor.py</code></li>
                            <li><code>mlp_regressor.py</code></li>
                            <li><code>regressors.py</code></li>
                            <li><code>svr_regressor.py</code></li>
                            <li><code>xgb_regressor.py</code></li>
                        </ul>
                    </details>
                </li>
            </ul>
        </details>
    </li>
    <li><strong>data.py</strong> - Script for loading and preprocessing data.</li>
    <li><strong>heuristics.py</strong> - Wrapper for metaheuristic optimization algorithms.</li>
    <li><strong>main.py</strong> - Main entry point for running the optimization experiments.</li>
    <li><strong>requirements.txt</strong> - Lists all Python dependencies for the project.</li>
    <li><strong>wavelet_ml_config.py</strong> - Defines the search space (models, parameters, wavelets).</li>
</ul>
</details>