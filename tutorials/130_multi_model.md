# Design Options

1. We create reference reader, create reference model, create scenario reader, 
update reference model with scenario reader. **SELECTED**
   * Pros: 
     * Allows us to save & re-use reference reader between scenarios.
     * Allows us to save & re-use reference model between scenarios. 
     * Doesn't require updates to Reader()
     * Doesn't require updates to Model()
   * Cons: 
     * 2 extra calls
     * Requires a new Model.update() method

2. We create reference reader, create scenario reader, create model(reference_reader, 
   scenario_reader). **REJECTED** 
   * Pros: 
     * Allows us to save & re-use reference reader between scenarios.
     * Doesn't require updates to Reader()
     * Doesn't require a new Model.update() method
   * Cons: 
     * 1 extra call.
     * Requires updates to Model()
     * Doesn't allow us to save & re-use reference model between scenarios [!]

3. We create a reference reader, create reference model, create scenario reader, create scenario
   model, update reference model with scenario model. **REJECTED**
   * Pros: 
     * Allows us to save & re-use reference reader between scenarios.
     * Allows us to save & re-use reference model between scenarios. 
     * Doesn't require updates to Reader()
     * Doesn't require updates to Model()
   * Cons: 
     * 3 extra calls. 
     * Requires a new Model.update() method
   
4. We create a reader(reference, scenario), create model. **REJECTED**
   * Pros: 
     * No additional calls. 
   * Cons: 
       * Doesn't allow us to save & re-use reference reader between scenarios.
       * Doesn't allow us to save & re-use reference model between scenarios. 
       * Requires updates to Reader()
       * Requires updates to Model()
       * Requires a new Model.update() method
       * Requires ensuring consistency between reference & scenario model 
         descriptions (in-terms of the reader parameters)


## Decision
* Rejected option 4 b/c too many cons without enough benefits
* Rejected option 2 b/c it doesn't allow us to re-use the model (model building
  takes the most amount of time)
* Rejected option 3 b/c of the additional calls, without any additional benefits
  over option 1
* Selected option 1

# Option 1 Implementation Notes
**Decision &mdash; We create reference reader, create reference model, create
scenario reader, update reference model with scenario reader.**

## 1. Create Reference Reader
Nothing new needs to be implemented here.

## 2. Create Reference Model
Nothing new needs to be implemented here. 

## 3. Create Scenario Reader
Nothing new needs to be implemented here. 

## 4. Update Reference Model with Scenario Reader
I will need to create an `Model.update()` method. This method will need to: 
* Iterate through each of the nodes in the scenario model and add them to the 
model if they don't already exist. If they do exist, it needs to update the
nodes parameters to match what is in the scenario
* Iterate through each of the technologies and add them to the model if they
don't already exist. If they do exist, it needs to update the tech's parameters
to match what is in the scenario. 

### Questions
* Do we need to follow this up by re-initializing Model parameters?
  * Currently, `Model()` calls `build_graph()`, `_dcc_classes()`,
  `_inherit_parameter_values()`, `_initialize_model()`
  * It also sets `self.node_dfs`, `self.tech_dfs`, `self.node_tech_defaults`, 
  `self.years`, `self.model_description_file`.
* Does it need to deal with new defaults?
* Idea: We keep the node_df dictionary. Is there someway we can use this to our
advantage? We could add the node. And then de-duplicate the DF by a subset of
columns, keeping the last instance. Then we could build the model from this.
Unfortunately, that doesn't really take advantage of the work already done to
build the reference model. So probably not. _However_, it could work for
updating `self.node_dfs` and `self.tech_dfs`. 