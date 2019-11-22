Qs:
- Is there data/values/anything associated with things above tech level?
- What 'owns' the computation, what the computation work on, how does the computation works. Do techs with the same name have the same
Do we know something about the attribute schema based on the tech name

What needs to 'contain' data? Do Sector, or Region conrtain data or are just labels?

- (what about elasticity/v?)

- Note: see p.49 of CIMS manual, mentions of differences between regions


some regions don't all the sectors

need to have trade betwen regions

calculate the price from supply and demand - some of them province wide (electricity)
some country wide (natural gas/oil biofuels) (put province/country flags)


make a function fast `mini-optimization` for the supply/demand (5% threshold). maybe VI?





classes:

- contains the whole thing

- tech class:
  at least 3 fields (region, sector, end use)
  name of tech
  contains a dictionary of attributes


- may need classes for types of attributes


formula classes: figure out how to map the formulas to the things we have


- write container class, takes as a constructor the json file

###

size - ditch


-----

#### To Do:

- [x] remove size/weighted values - not necessary, can be computed separately

#### Early May:

- [ ] Need way to convert xml to json in a clean manner, if possible
  (look in to [this](https://stackoverflow.com/questions/191536/converting-xml-to-json-using-python))

- [ ] looking into way to retain the tree information (iterate through nested dict properly)

- [ ] trying to figure out how to generalize the structure

- [ ] Need to build a system to keep track where in the tree/graph we are (becomes complicated with the bidirectional edges created by supply side)

#### Notes
*Nov 20, 2019*
* Information (like prices) should be inherited from parent nodes. This is where we talked about if a data is needed and 
isn't available at the node itself, look to the ancestors for the information. **Question: What happens when a node has
multiple parents? Who should it inherit from? Is this based on structure which stipulates 1 parent per node?**

*Nov 13, 2019*
* JA implemented an abstract aggregation function that adds fuels demand from each node. BG asked that the aggregation 
function be made abstract enough to be able to aggregate other attributes as well. JA: confident that this will be able
to work, but for now we can operate as is and abstract/generalize during software engineering phase. 

* BG: Service cost has been removed from the model description since all of its components have already been provided in
the description. By avoiding the repetitive and manual defining of equations, ideally we can avoid mistakes. The service
cost calculation will be added to the model description soon, in the Formulas sheet. This calcualtion will only need to
be done for services & technologies during the intial phase. And the calculation of service cost becomes a leaf->root 
calculation. 

* BG: Looking ahead, we will need to keep track of emissions for technologies and services. Hopefully have this as a 
factor where energy for a fuel is mapped to emission. Then we will be given a table with the fuel to emission 
calculations as well. 

* BG: There are some model attributes that will need to exist, but won't be defined in the model description. 
E.g. Stock or Life Cycle Cost. We will rely on python for the calculations of these values instead. BG&RS are going to 
get a list of these attributes to ML&JA. 

* Refer to Technologies table for default values for technology and service parameters. These defaults are all scalars
 and will not be changed or calculated over time. 

* In the model description, unchanging values have now been filled in. This means that empty cells indicate a calue that
needs to be calculated.


