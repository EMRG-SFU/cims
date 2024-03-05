# Getting Started


Once the `CIMS` Python package is installed ([instructions here](INSTALL.md)), 
you are ready to run economic-climate models using `CIMS`. If you are new to 
`CIMS`, we suggest following the 
[Quickstart Jupyter Notebook Tutorial](../tutorials/Quickstart.ipynb). If you'd
prefer to avoid Jupyter Notebooks, you can also follow our 
[CIMS_basic.py](../tutorials/CIMS_basic.py) script, which provides a minimal
code example for how to use `CIMS`.

Regardless of how you choose to use `CIMS`, you will need a model file to use as
input. Example model files can be found on the 
[`cims-models`](https://github.com/EMRG-SFU/cims-models/) repository. You can
access these model files in two ways:
1. If you installed the `CIMS` Python package by following the 
[cims-meta](https://github.com/EMRG-SFU/cims-meta) repository instructions, 
compatible model files will already have been downloaded for you. You will find
these inside the `cims-models` directory which would have been downloaded into 
the `cims-meta` directory when you installed `CIMS`.
2. Alternatively, you can directly clone the 
[`cims-models` repository](https://github.com/EMRG-SFU/cims-models/) from
GitHub. 
    ```shell
    $ git clone https://github.com/EMRG-SFU/cims-models/
    ```
    This will download a directory containing sample model descriptions which
    you can use as-is or customize for your own modelling needs.
    > [!WARNING] If you are using an older version of `CIMS`, you may find the model files downloaded when cloning the latest version of the `cims-model` repository are incompatible with your `CIMS` version. 
    >
    > For this reason, we recommend installing CIMS via the [`cims-meta` repository](https://github.com/EMRG-SFU/cims-meta).   
    > 
    > If this remains an infeasible option, review the version-specific [CIMS release notes](https://github.com/EMRG-SFU/cims/releases) to find a compatible `cims-model` release. Then, use the version number to download compatible model files: `git clone --branch v1.0 https://github.com/EMRG-SFU/cims-models/@`)
