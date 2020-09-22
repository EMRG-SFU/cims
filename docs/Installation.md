# Installation & Setup


## Clone the Repo
### 1. Open the command line.    
On Linux or Mac machines, open Terminal (pre-installed). On Windows, open GitBash. If you do not have Git Bash installed, you can install it from [here](https://gitforwindows.org/). Other command line programs such as Bash can work, but may involve extra configuring.   

### 2. Check for an existing SSH key pair   
Follow GitHub’s [instructions](https://docs.github.com/en/github/authenticating-to-github/checking-for-existing-ssh-keys) to check whether your computer has an existing SSH key pair. Be sure to follow the instructions specific to your machine’s operating system (Mac, Windows, Linux).

If your computer has an SSH key pair which you’d like to use you can skip straight to step 4. Otherwise, continue onto step 3.  

### 3. Generate a new SSH key pair
Follow GitHub’s [instructions](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to generate a new key pair. The instructions regarding “Adding your SSH key to the ssh-agent” are optional. Be sure to follow the instructions specific to your machine’s operating system (Mac, Windows, Linux).  

### 4. Upload the public SSH key to your GitLab account
Now you can copy the SSH key you created to your GitLab account. To do so, follow these steps:

* Copy your public SSH key to a location that saves information in text format. The following options saves information for ED25519 keys to the clipboard for the noted operating system:   
    **macOS:**
    ```
    pbcopy < ~/.ssh/id_ed25519.pub
    ```
    **Linux (requires the xclip package):**
    ```
    xclip -sel clip < ~/.ssh/id_ed25519.pub
    ```
    **Git Bash on Windows:**
    ```
    cat ~/.ssh/id_ed25519.pub | clip
    ```

* If you’re using an RSA key, substitute accordingly.
* Navigate to https://gitlab.com and sign in.
* Select your avatar in the upper right corner, and click Settings
* Click SSH Keys.
* Paste the public key that you copied into the Key text box.
* Make sure your key includes a descriptive name in the Title text box, such as Work Laptop or Home Workstation.
* Include an (optional) expiry date for the key under “Expires at” section. (Introduced in GitLab 12.9.)
* Click the Add key button. 

>Note: If you manually copied your public SSH key make sure you copied the entire key starting with ssh-ed25519 (or ssh-rsa) and ending with your email address.

### 5. Clone the Git Repository using SSH
* Open Terminal or Git Bash (depending on your operating system)
* Change the current working directory to the location where you want the cloned directory to be made. 
* Paste the following command into the command line
```
git clone git@gitlab.rcg.sfu.ca:mlachain/pycims_prototype.git
```
* Press Enter. Your local clone will be created. 
* Navigate into the directory using the following command:
```
cd pycims_prototype
```

## Install pyCIMS
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
