### Summary August 2019

- **Problem from previous meeting:** The system was not equipped to differentiate between techs competing for fuel and techs with different fuel requirements. This lead to nonsensical results as independent techs were splitting the market shares together. In the future, having multiple sectors may compound this problem.
<br/>The data structure proposed in the last meeting was very simple and was not hierarchical. Arrays of size $n$ were used to model $n$ technologies, with the names indexed in a separate array. This did not allow for the label of special relationships between technologies in terms of competition without a major overhaul.

- **Solutions proposed:** There is a trade-off between generalization and clarity for this problem. We want to avoid re-designing the algorithm for each individual problem. To identify any mistakes or corner cases, and to communicate effectively between fields (between programmer and researcher), a clearly presented, non generalized version of the problem is much better. It would also be useful for use in python tutorials for future students who may use CIMS. We want to avoid spending too much time generalizing prior to knowing future issues with the model, but rewriting the problem with each modification is also time consuming.
  - **Solution 1 (short term)** Restructure the data to include some level of hierarchy
  - **Solution 2 (long term)** Object-path for easier path to data attributes





 (aggregation issue within differnet sectors)
