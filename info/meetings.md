## Meeting Agendas, pyCIMS project

---

### May 27th, 2019 - Brad, Steven and Maude

Brad and Steven: Before the meeting, if you get a chance to check the issues in the CIMS-docs repository, please do it. Otherwise, we'll take a quick look - most of these issues will be covered below.

- [ ] Define exogenous 'Demand' input variables - if we need to make up values, get data their data structure.
- [ ] Make plans to match words in simplified model and full model and documentation
- [ ] Cleaned up dataset check
- [ ] Discuss Docs written so far
- [ ] Plans to identify operating equations (probably linked to writing of docs)
- [ ] Specific: check bet/new stock retirement calculations
- [ ] What related to `Assessment of Capital Stock Availability` in docs (one of the 6 components of the model)


---

#### May 14th 2019

9:30am, Steven & Maude

### From previous conversation:

- [x] Need way to convert xml to json in a clean manner
  (look in to [this](https://stackoverflow.com/questions/191536/converting-xml-to-json-using-python))

- [x] looking into way to retain the tree information (iterate through nested dict properly) *done with data cleaning code, might need adjustments later*

- [x] code will be generalized iteratively, some things are too specific atm

- [x] Need to build a system to keep track where in the tree/graph we are (becomes complicated with the bidirectional edges created by supply side) - *basically done in data cleaning script, but could be improved*


### To Discuss:

- [x] Set next meeting with Brad, hopefully all 3 of us (Friday the 24th or Monday the 27th).

### pyCIMS:
- [x] Defining what a simple model looks like *in progress*

- [x] Dealing with inconsistent information
  -  we'll probably need to dig into the java/APL code, what is our plan
  **solution**: Maybe a CS RA can speed up the process


- [x] What form the input/output should take
  - Right now the input is a giant xml file with a lot of unimportant info,
    also field names are not straightforward (many single letters)
  - JCIMS currently outputs almost 200 files in 17 different directories, making it difficult to simplify
  - Determine all inputs to make up (eg: Demand)
  _**In progress**_

##### Main Issue: Relationship between nodes are often unknown, and the model, as documented, is incomplete and doesn't have a 'core'

(the definition of a node is not exact either, as it means different things throughout the docs)

- [x] Where is the model definition contained
  - In JCIMS, it is fully contained inside the xml file
  - It needs to be changeable by the user
  - Brad would like to have the input data be a excel table that translates into a
    json file, to be read as a nested dictionary in python
  - Model definition can be specified in separate file

---
