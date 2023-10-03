# Working with the CIMS Conda Environment

If you followed the [installation steps](INSTALL.md) for CIMS, you may have noticed that we used a Conda environment. This environment allows us to use and manage Python versions, packages, and tools without affecting the global environment on your computer. The `CIMS_env` environment we created includes all the dependencies needed to run CIMS, including the CIMS package itself. 

To gain access to this environment, we need to "activate" it. This activation is what will allow us to use and run CIMS. 

Below, you will find instructions for how to activate the environment using the command line. This is needed to run Jupyter Notebooks, such as the [Quickstart](../tutorials/Quickstart.ipynb) and [ModelValidation](../tutorials/ModelValidation.ipynb) guides. 

Alternatively, if you use IDEs such as Spyder, PyCharm, or Atom you may need to activate the environment in another way. You can find helpful links regarding how to do this at the end of this document. 

## From the commandline

### Activate Environment
If you run CIMS from outside an IDE such as PyCharm or VSCode, you'll need to activate your `cims` environment each time you open a new Terminal or Git Bash window.
Luckily, activating an environment isn't hard. From the commandline (Git Bash or Terminal), run the following: 
```
conda activate cims
```
You do not need to be in a particular directory for this to work.
You should now see `(cims)` at the start of your commandline prompt. This indicates that the environment was correctly activated.
You can now import CIMS into your Python notebooks and scripts. 

### De-Activate Environment
From the command line, run the following command to de-activate the CIMS conda environment. 
```
conda deactivate
```
You should no longer see `(cims)` at the start of your commandline prompt. This indicates that the environment was correctly de-activated. 

## Activating Conda Environments in IDEs
Depending on the development tools you use, activating the `cims` environment from the command line might not actually make the environment available where you write and run your code. I've included links for some of the most popular IDEs below.
* PyCharm - https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/
* VS Code - https://code.visualstudio.com/docs/python/environments#_create-a-conda-environment-in-the-terminal
* Atom - https://stackoverflow.com/questions/43207427/using-anaconda-environment-in-atom
* Spyder - https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment
