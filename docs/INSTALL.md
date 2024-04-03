# Installing CIMS

## Requirements
Before you can install CIMS, you'll need to ensure bash, git, and Python are installed on your computer. 

If you're new to programming and managing software installations through the commandline, we suggest following the installation instructions found on the [Software Carpentries]() website. Specifically, look for these 3 sections (Be sure to use the installation instructions which correspond to your operating system):
* [bash shell](https://carpentries.github.io/workshop-template/install_instructions/#the-bash-shell)
* [git](https://carpentries.github.io/workshop-template/install_instructions/#git-1)
* [Python](https://carpentries.github.io/workshop-template/install_instructions/#python-1)


> [!NOTE]  
> If you're already comfortable maintaining and managing software installations outside of Anaconda, go ahead and use the workflow you're used to.

## Installation
### Basic Installation
1. **Setup a virtual environment** _(optional)_   
   If you are using a virtual environment, but are unfamiliar with how to use them, follow [these instructions](#using-virtual-environments) to create and activate your virtual environment.

2. **Retrieve your GitHub PAT**   
   If you are installing `CIMS` prior to its public release, you will need to use a Personal Access Token (PAT) from GitHub to do so. For instructions on creating these tokens, checkout [GitHub's documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens). 
   
   Once you have your PAT, open the `install-cims.sh` file in a text editor and update line 8 with your own PAT from GitHub.

3. **Install `CIMS`**  
   Replacing `TOKEN` in the following command with the `PAT` you retreived in
   step 2, install `CIMS` using pip.
   ```shell
   $ pip3 install git+https://TOKEN@github.com/EMRG-SFU/cims.git
   ```

   If you want to install a specific version of `CIMS` (e.g. `v1.0.0`) you can
   use the following command. Be sure to use the relevant version name (found at
   the end of the command).
   ```shell
   pip3 install git+https://TOKEN@github.com/EMRG-SFU/cims.git@v1.0.0
   ``` 
   ``
4. **Confirm the Installation Succeeded**
   To ensure the Python package has been installed, open a Python Shell and try
   to import the `CIMS` package.
      ```shell
      $ python3
         >>> import CIMS
      ```   
      If this completes without raising errors or warnings, then you are good to go!

### Advanced Installation
In certain situations, the [basic installation](#basic-installation)
instructions will not be sufficient. For example, if you are contributing to 
`CIMS` Python development you'll want to see local code changes reflected in
your installed version of `CIMS`. In these cases, it may make sense to follow
these advanced installation instructions. 

1. **Clone the GitHub repository**
   ```shell
   $ git clone https://github.com/EMRG-SFU/cims.git
   ```

2. **Navigate to inside the `cims` directory**
   ```shell
   $ cd cims
   ```

3. **Setup a virtual environment** _(optional)_   
   If you are using a virtual environment, but are unfamiliar with how to use them, follow [these instructions](#using-virtual-environments) to create and activate your virtual environment.

4. **Install CIMS**
   Using `pip` install `CIMS` in development mode from source. This will ensure any changes you make to the source code are reflected in the `CIMS` package.
   ```shell
   $ pip install -e .
   ```
   
5. **Confirm the Installation Succeeded**
   To ensure the Python package has been installed, open a Python Shell and try
   to import the `CIMS` package.
      ```shell
      $ python3
         >>> import CIMS
      ```   
      If this completes without raising errors or warnings, then you are good to go!


## Using Virtual Environments
More information on working with Python Virtual Environments can be read [here](https://realpython.com/python-virtual-environments-a-primer/).   

> [!NOTE]  
> Anaconda offers an alternative to virtual environments. Feel free to use it instead.


### Create a Virtual Environment
Create a brand new virtual environment using `venv` passing in a filepath to specify where the virtual environment will be saved.
```shell
$ python -m venv /path/to/new/virtual/environment
```

For example, the following command would create a virtual environment called `cims-venv` in your current directory.
```shell
$ python -m venv ./cims-venv 
```

### Activate your Virtual Environment
Before you can use your virtual environment, you need to activate it. For example, 

* On Windows
   ```shell
   $  .\cims-venv\Scripts\activate
   ```
* On Mac/Linux
   ```shell
   $ source ./cims-venv/bin/activate 
   ```

Once your virtual environment is successfully activated, your command prompt will change to include the virtual environment's name. For example: 
```shell
(cims-venv) $
```

Any Python packages you install while the virtual environment is activated will be available _inside_ the virtual environment but not outside of it. This allows you to install specific package versions for a single project, without impacting the packages you have available for other projects. 

Note, you only need to create & setup your virtual environment once. However, you do need to activate your virtual environment each time you want to use it.   

### Deactivate your Virtual Environment
Once you're done using your virtual environment (or if you want to change to a different virtual environment), you can de-activate the environment. 
```shell
(cims-venv) $ deactivate
```
Your command prompt will return to normal (with the environment name having disappeared) and the packages available within the virtual environment will no longer be available to you (unless they are also installed globally on your machine)
```shell
$
```