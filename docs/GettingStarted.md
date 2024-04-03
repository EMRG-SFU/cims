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
1. If you've used the `download()` available through `CIMS` you will have already downloaded models compatible with the installed version of `CIMS`.
   
2. Alternatively, you can directly clone the 
[`cims-models` repository](https://github.com/EMRG-SFU/cims-models/) from
GitHub. 
    ```shell
    $ git clone https://github.com/EMRG-SFU/cims-models/
    ```
    This will download a directory containing sample model descriptions which
    you can use as-is or customize for your own modelling needs.

> [!CAUTION]
> If you are using an older version of `CIMS`, you may find the model files obtained through method 2 are incompatible with your `CIMS` version. 

