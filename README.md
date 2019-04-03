<hr/>

# Prototype for modernization and customization of [CIMS](https://pics.uvic.ca/sites/default/files/uploads/CIMS%20Community%20Excel%20model%20user%20documentation_0.pdf) 

_using Python 3.7.2_

_**Spring/Summer 2019, Big Data Hub, SFU**_


<hr/>

### Prototyping, brainstorming, and designing an updated, modularized and customizable version of ISTUM/CIMS, a realistic technology simulation model. 

_See [overview](#view) and demos for details._


[![Follow](https://img.shields.io/twitter/follow/maude_ll.svg?style=social&label=Follow)](https://twitter.com/maude_ll)


## :clipboard: Table of Contents

* [Overview](#view)

* [Environment](#env)

* [Launching a Demo](#launch)

* [Data description](#data)

* [Simulation Structure Overview](#sim)
   * [Tree Structure](#tree_strcu)
   * [Node Types](#node)
   * [Equations and Connection Rules](#eqn)
   * [Additional Info](#add_infp)

* [Algorithms](#alg)
  * [Simple Pseudocode and other Recipes](#pseudocode)
  * [The Fancier Route - Time and Space](#fancy)
  * [Calibration](#calib)

* [Contact Info](#fin)

<hr/>

## :eye: Overview <a name="view"></a>
CIMS: simulation models for economy, energy and materials

  - Goal: understand relationship between energy policy and firms/households behaviour w.r.t. tech acquisition and use.
  - Designed to help policymakers take decisions - descriptive rather than normative

  - Gives information on:
    - how/what policy changes can lead to specific political objectives
    - explicit and implicit costs of specific policies
    - development of a model/simulation with uncertainty quantification about the inputs
    - energy flows through economic system, from production to individual technology use (global and local)
    - useful to model policy on energy efficiency, greenhouse gas and air quality
    - fuel consumption, emission estimates approximated by economic activity

  - CIMS works at the macro and micro level simultaneously, as an (almost) full equilibrium model


Written for Dr. Bradford Griffin by Big Data Hub RA Maude Lachaine under the supervision of Dr. Steven Bergner.

<hr/>

## :seedling: Environment <a name="env"></a>

All demos are implemented in python 3.7.2 If using [Pipenv](https://pipenv.readthedocs.io/en/latest/), use the Pipfile to set up your environment. If you prefer not using Pipenv, use the requirement.txt file to set up an environment or for documentation about working library versions and requirements.

## :rocket: Launching a Demo <a name="launch"></a>

**Note:** The notebooks were intended to be viewed using the [`jupyterthemes` package](https://github.com/dunovank/jupyter-themes).

Once in the proper environment, type on the CL:

```unix
jt -t grade3 -f anonymous -fs 13 -cellw 100% -T -N
```

* **Running a notebook:** start an environment from a shell, then type:

```unix

jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

```

The flag may be necessary to view some interactive plots, depending on your machine.



## :bar_chart: Data description <a name="data"></a>

_To be completed: description of xml file, transition to json to cson, and the tree structure encoded in file_

## :hammer: Simulation Structure Overview <a name="sim"></a>

For details on methods and implemention, view the notebook directory.

_Branches below tbd_

### Tree Structure <a name="tree_strcu"></a>

<br/>
[...]

### Node Types<a name="node"></a>

<br/>
[...]


### Equations and Connection Rules<a name="eqn"></a>

<br/>
[...]

### Additional Info<a name="add_infp"></a>

<br/>
[...]

<hr/>
<hr/>

###  :space_invader: Contact Info <a name="fin"></a>

For questions or details, contact me at `mlachain (at) sfu (dot) ca`
