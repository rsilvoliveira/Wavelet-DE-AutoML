# Wavelet-DE-AutoML

A novel automated framework for accurate streamflow forecasting that integrates wavelet transforms, Differential Evolution optimization, and Automated Machine Learning.

## Description

**Wavelet-DE-AutoML** is an automated framework for streamflow forecasting that integrates *wavelet transform-based preprocessing*, *Differential Evolution optimization*, and *Automated Machine Learning (AutoML)*.  
It simultaneously optimizes wavelet parameters, input lags, model selection, and hyperparameters to build high-performance forecasting models across multiple temporal scales (hourly, daily, and monthly).  
While the framework supports multiple metaheuristic algorithms for optimization, its primary focus is on the Differential Evolution algorithm, which demonstrated superior efficiency and accuracy in systematic evaluations.  
By combining wavelet-based feature extraction with metaheuristic optimization, the framework achieves superior accuracy and robustness compared to conventional AutoML baselines, providing a scalable and interpretable solution for hydrological forecasting and water-resource management.



## How It Works

This framework is designed to automate the process of building high-performance time series forecasting models. It takes a raw time series as input and uses a Differential Evolution metaheuristic to explore a vast search space of possible modeling pipelines.

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

* [Auto-sklearn Installation Guide](https://automl.github.io/auto-sklearn/master/installation.html)
* [AutoKeras Installation Guide](https://autokeras.com/install/)

> [!NOTE]
> These libraries have specific dependencies (e.g., Python versions, build tools, ML frameworks) that may differ from the core requirements of this project. Please review their documentation carefully before proceeding with the installation.


## Configuration and Execution

To execute the code, simply run the `main.py` script:
```
python main.py
```
The main parameters can be modified directly in the `main.py` file, inside the following block at the end of the script:
```
if __name__ == "__main__":
```
Below is a description of the configurable parameters:

* **RUNS (`int`)**: Number of experiment repetitions.

* **FORECAST (`list[int]`)**: Forecast horizons to be used (depending on `TIME_SCALE`, e.g., in hours, days, or months).

* **DATABASE (`dict`)**: Data sources to be used in the experiment.

* **TIME_SCALE (`str`)**: Time resolution of the data. Available options:

    * "hourly"

    * "daily"

    * "monthly"

* **heuristics (`list[str]`)**: List of metaheuristic algorithms to be executed (e.g., `"de"`, `"pso"`, etc.).

* **POPULATION (`int`)**: Population size used by the optimization algorithms.

* **ITERACTIONS (`int`)**: Number of iterations for the optimization process.

* **TARGET_VALUE (`float` or `int`)**:  
  Early stopping threshold for the optimization process.  
  The heuristic evaluates model performance using the MAPE (Mean Absolute Percentage Error).  
  If the MAPE falls below this target value, the algorithm stops early.

* **N_JOBS (`int`)**: Number of parallel jobs to be used (for multiprocessing).

Example:
```
RUNS = 3
FORECAST = [1, 3, 7]
DATABASE = {"stations": ["58880001", "58235100"]}
TIME_SCALE = "daily"
heuristics = ["de"]
POPULATION = 70
ITERACTIONS = 150
TARGET_VALUE = 1
N_JOBS = 3
```
> [!NOTE]
> Some options are commented out by default and can be enabled by uncommenting them as needed.

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
