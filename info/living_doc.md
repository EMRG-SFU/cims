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
