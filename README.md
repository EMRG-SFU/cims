# CIMS
CIMS is a Python package providing the Python implementation of the [CIMS](https://pics.uvic.ca/sites/default/files/uploads/CIMS%20Community%20Excel%20model%20user%20documentation_0.pdf)
economic climate model. 

## :gear: Installation
CIMS is not currently available on PyPi or other package indexes. Follow the [installation guide](docs/Installation.md) to get CIMS running on your machine.

## :technologist: Usage
Once you've installed CIMS you can call its functions and classes from within your own Python script, notebook, or program. Follow the [quickstart guide](tutorials/Quickstart.ipynb) to familiarize yourself with CIMS's key functionality. 

```
import CIMS
model_file = 'path/to/model.xlsb'
my_reader = CIMS.ModelReader(infile=model_file)
my_model = CIMS.Model(my_reader)
my_model.run()
```
## :memo: Contributing
Contributions to CIMS are welcome, in many different forms: 
* **Issues** &mdash; If you identify a bug, error in the documentation, or a potential improvement to CIMS, consider putting this information into an issue. First, search the list of [existing issues](https://github.com/EMRG-SFU/cims/issues) to see if there is an ongoing discussion to join. If a relevant issue doesn't already exist, please [create a new issue](https://github.com/EMRG-SFU/cims/issues/new).
* **Code** &mdash; If you are comfortable writing code feel free to make a Pull Request (PR) with your changes. If you've tackled a large feature request or bug, please also create a new issue, or mention an existing issue within your PR.
* **Documentation** &mdash; If you notice typos, out-of-date information, or opportunities for improvements in the documentation (and are comfortable writing [Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)), please consider making a PR with changes.

Any kind of contribution, whether its fixing a small typo, refactoring existing code, or the implementation of a brand new module helps improve this project. 

## :book: Citation

## :pray: Acknowledgements 
[Bradford Griffin](https://github.com/brad-griffin) and [Jillian Anderson](https://github.com/jillianderson8) are the project's lead researcher and lead technical developer, respectively.

In addition, contributions to the codebase have been made by members of Simon Fraser University's [Big Data Hub](https://www.sfu.ca/big-data.html) and [Research Computing Group](https://www.rcg.sfu.ca/): 
* [Steven Bergner](https://github.com/git-steb)
* [Rashid Barket](https://github.com/rbarket)
* [Maude Lachaine](https://github.com/semaphore-maude)
* [Adriena Wong](https://github.com/atwong88)
* Daisy Yu
* Kacy Wu


Finally, thank you to the numerous EMRG graduate students who have attended meetings, submitted features requests, and flagged bugs: 
* [Thomas Budd](https://github.com/tcbudd)
* Aaron Pardy
* Emma Starke
* Kaitlin Thompson
* Heather Chambers
* Ryan Safton

## :balance_scale: License 
The CIMS Python library is licensed under the [MIT License](https://github.com/EMRG-SFU/cims/blob/main/LICENSE). For more information about this license, checkout [this overview](https://choosealicense.com/licenses/mit/).
