# Notes
A place to house notes and answers to questions, primarily those that have come up in meetings. Currently, they are
sorted by date, but that isn't super important and will likely make sense to change in the future. 

### Nov 20, 2019
* Information (like prices) should be inherited from parent nodes. This is where we talked about if a data is needed and 
isn't available at the node itself, look to the ancestors for the information. **Question: What happens when a node has
multiple parents? Who should it inherit from? Is this based on structure which stipulates 1 parent per node?**

### Nov 13, 2019
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


