# Updating CIMS

These instructions assume that you have installed CIMS according to the [installation instructions](Installation.md). 

Using your Bash shell (Git Bash on Windows, Terminal on Linux or MacOS) upgrade CIMS.  

1. **Activate the CIMS conda environment**
```
conda activate CIMS_env
```   

2. **Navigate to your local version of the CIMS repository**
```
cd PATH/TO/FOLDER
```   

3. **Update the repositry**
```
git pull
```
    You may encounter a "merge conflict" if you have made local changes to the CIMS code. If this is your first time encountering a merge confict in git (or you want a refresher), I suggest reading through the Git Tower article  on [dealing with merge requests](https://www.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/merge-conflicts/). 
    
    Merge conflicts can be particularly messy if you are working with Jupyter Notebooks. If you want to resolve merge conflicts in Jupyter notebooks (rather than overwrite your local changes), I would suggest using [nbdime](https://nbdime.readthedocs.io/en/latest/#). There is both a command line and web tool that simplifies merge conflict resolution. 


4. **Restart active kernels**   
    Restart any active kernels that are running the old version of CIMS. If you are using Jupyter Notebook, do this by selecting one of the restart (e.g. Restart, Restart & Clear Output, etc) options under the "Kernel" menu at the top of the notebook. In other IDEs such as Spyder, PyCharm, or Atom this usually requires a reload of the Python Console. 