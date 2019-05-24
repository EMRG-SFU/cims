#### May 14th 2019

9:30am, Steven & Maude

### From previous conversation:

- [ ] Need way to convert xml to json in a clean manner, if possible
  (look in to [this](https://stackoverflow.com/questions/191536/converting-xml-to-json-using-python))

- [ ] looking into way to retain the tree information (iterate through nested dict properly)

- [ ] trying to figure out how to generalize the structure

- [ ] Need to build a system to keep track where in the tree/graph we are (becomes complicated with the bidirectional edges created by supply side)


### To Discuss:

- [ ] Set next meeting with Brad, hopefully all 3 of us (Friday the 24th or Monday the 27th).

### pyCIMS:
- [ ] Defining what a simple model looks like

- [ ] Dealing with inconsistent information
  - Brad figures we'll need to dig into the java/APL code, what is our plan


- [ ] What form the input/output should take
  - Right now the input is a giant xml file with a lot of unimportant info,
    also field names are not straightforward (many single letters)
  - JCIMS currently outputs almost 200 files in 17 different directories, making it difficult to simplify
  - Determine all inputs to make up

##### Main Issue: Relationship between nodes are often unknown, and the model, as documented, is incomplete and doesn't have a 'core'

(the concept of a node is not clear either, as it means different things throughout the docs)

- [ ] Where is the model definition contained
  - In JCIMS, it is fully contained inside the xml file
  - It needs to be changeable by the user
  - Brad would like to have the input data be a excel table that translates into a
    json file, to be read as a nested dictionary in python
  - Model definition can be specified in separate file
