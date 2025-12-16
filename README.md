# SkySeaLand Detector

TODO: Pretty picture

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


## Dataset Download

To download the data, first ensure that you performed the [developer install](#developer-install), then run:
```
skysealand download
```

Ensure that a `data` directory has been created in the root directory and that it is not empty.

This also checks for any issues with the downloaded data. To confirm that everything was correctly downloaded, inspect the newly created `validation_report.json` in the downloaded data folder.


## Training the model

To train the baseline version of the model, first ensure that you performed the [developer install](#developer-install) and have [downloaded the dataset](#dataset-download), then run:
```
skysealand train
```

Then you can check the output of this in the `runs` directory.
