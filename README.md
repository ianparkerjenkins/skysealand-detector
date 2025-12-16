# SkySeaLand Detector


## Developer Install

Close the repository
```
git clone https://github.com/ianparkerjenkins/skysealand-detector.git
```

Navigate into the repository
```
cd skysealand-detector
```

Install mamba (if not already installed)
```
conda install -c conda-forge mamba
```

Create the conda environment.
```
mambda env create -f dev_environment.yml
```

Activate the new environment.
```
conda activate skysealand
```

Run the `devinstall.py` script.
```
python devinstall.py
```

Launch VSCode.
```
code .
```

## Training the model

### Dataset Download

TODO: Convert these to typer CLI commands

Download the data
```
python ./src/skysealand/dataset/download.py
```

Ensure that a `data` directory has been created in the root directory and that it is not empty.

Validate that the data was downloaded correctly
```
python ./src/skysealand/dataset/validation.py
```

Then inspect the newly created `validation_report.json` in the downloaded data folder.
