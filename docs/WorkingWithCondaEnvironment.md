# Working with the CIMS Conda Environment

If you followed the [installation steps](Installation.md) for CIMS, you may have noticed that we used a Conda environment. This environment allows us to use and manage Python versions, packages, and tools without affecting the global environment on your computer. The `CIMS_env` environment we created includes all of the dependencies needed to run CIMS, including the CIMS package itself. 

To gain access to this environment, we need to "activate" it. This activation is what will allow us to use and run CIMS. 

Below, you will find instructions for how to activate the evironment using the command line. This is needed to run Jupyter Notebooks, such as the [Quickstart](../code/tutorials_examples/Quickstart.ipynb) and [ModelValidation](../code/tutorials_examples/ModelValidation.ipynb) guides. 

Alternatively, if you use IDEs such as Spyder, PyCharm, or Atom you may need to activate the environment in another way. You can find helpful links regarding how to do this at the end of this document. 

## Command Line Activation
The following 3 steps discuss how you can activate (and later deactivate) the `CIMS_env` environment on the command line. If you use the command line (including Jupyter Notebooks) to run CIMS code, this activation will need to be done everytime you open a new Terminal or Bash window.  

### 1. Open the command line.    
On Linux or Mac machines, open Terminal (pre-installed). On Windows, open GitBash. If you do not have Git Bash installed, you can install it from [here](https://gitforwindows.org/). Other command line programs such as Bash can work, but may involve extra configuring.   


### 2. Activating the Environment

From the command line, run the following command to activate the CIMS conda environment. 
```
conda activate CIMS_env
```
You should now see `(CIMS_env)` prepended to your command line. This indicates that the environment was correctly activated. 

### 3. De-activating the Environment
From the command line, run the following command to de-activate the CIMS conda environment. 
```
conda deactivate
```
You should no longer see `(CIMS_env)` prepended to your command line. This indicates that the environment was correctly de-activated. 

## Other IDEs
Depending on the development tools you use, activating the `CIMS_env` environment from the command line might not actually make the environment available where you write and run your code. I've included links for some of the most popular IDEs below.
* Spyder - https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment
* PyCharm - https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/
* Atom - https://stackoverflow.com/questions/43207427/using-anaconda-environment-in-atom