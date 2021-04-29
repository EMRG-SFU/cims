# Installation & Setup

There are 3 steps for setting up pyCIMS on your personal computer. 
1. [Install required software](#required)
2. [Clone the git repository](#clone)
3. [Install pyCIMS](#pycims)


## Install Required Software <a id=required></a>
Before you install pyCIMS make sure you have 3 pieces of software installed on your computer: 
* Bash Shell
* Git
* Python

All three can be installed by following the instructions found on the Software Carpentries website. Specifically, follow the instructions under "The Bash Shell", "Git", and "Python" sections found [here](https://carpentries.github.io/workshop-template). Be sure to follow the instructions corresponding to your computer's operating system. 


## Clone the Repo<a id=clone></a>
1. **Create a Personal Access Token**
    * Log in to [GitLab](https://gitlab.rcg.sfu.ca/). 
    * In the upper-right corner, click your avatar and select **Settings**.
    * On the **User Settings** menu (left hand panel), select **Access Tokens**.
    * Choose a name and optional expiry date for the token.
    * Select the **write_repository** scope
    * Click the **Create personal access** token button.
    * Save the personal access token somewhere safe. Once you leave or refresh the page, you won't be able to access it again.
2. **Open the command line**
On Linux or Mac machines, open Terminal (pre-installed). On Windows, open GitBash. If you do not have Git Bash installed, you can install it from [here](https://gitforwindows.org/). Other command line programs such as Bash can work, but may involve extra configuring.
3. **Clone the Git Repository using HTTPS**
    * Change the current working directory to the location where you want the cloned directory to be made. 
    * Paste the following command into the command line
    ```
    git clone https://gitlab.rcg.sfu.ca/mlachain/pycims_prototype.git
    ```
    * You'll be prompted for a username and password. Your username is your SFU username (jdoe), not your SFU email (e.g. jdoe@sfu.ca). Your password is the Personal Access Token you created in step 1.
    * Navigate into the directory using the following command:
    ```
    cd pycims_prototype
    ```

## Install pyCIMS <a id=pycims></a>
1. From within the `pycims_prototype` directory, install dependencies using Anaconda. 
```
conda env create -f environment.yml python=3.6
```

2. Activate the conda environment.
```
conda activate pyCIMS_env
```

3. From within the `pycims_prototype` directory, install pyCIMS. 
```
pip install -e .
```
