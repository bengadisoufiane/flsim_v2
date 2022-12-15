# FLSim



## Installation

To install **FLSim**, all that needs to be done is clone this repository to the desired directory.

### Dependencies

**FLSim** uses [Anaconda](https://www.anaconda.com/distribution/) to manage Python and it's dependencies, listed in [`environment.yml`](environment.yml). To install the `fl-py37` Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:

```shell
conda env create -f environment.yml
```

## Usage

Before using the repository, make sure to activate the `fl-py37` environment with:

```shell
conda activate rl
```

### Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
python run.py
  --config=config.json
  --log=INFO
```

##### `run.py` flags

* `--config` (`-c`): path to the configuration file to be used.
* `--log` (`-l`): level of logging info to be written to console, defaults to `INFO`.
