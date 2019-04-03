WHAT IS CIMS:

  - (ML! ensembled?) simulation models for economy, energy and materials
  - Goal: understand relationship between energy policy and firms/households behaviour w.r.t. tech acquisition and use.
  - Designed to help policymakers take decisions - descriptive rather than normative

  - Gives info on:
    - how/what policy changes can lead to specific political objectives
    - costs of said policies
    - model/simulation uncertainty quantification (ML! really? :grimacing:)
    - energy flows through economic system, from production to individual tech use
    - useful to model policy on energy efficiency, greenhouse gas/air quality
    - (note: fuel consumption, emission estimates approx by economic activity)

  - works at the macro/micro level, (almost) full equilibrium model




<hr/>

**ISTUM**: individual subcomponents (nodes?) of CIMS.

  - puts techs against one another to serve pre-set user demand
  - only works at micro level, partial equilibrium




 ### Simulation

  - Engineering information: energy consumption of each energy service technology and shows how energy supply and demand are related.

  - Economic information: what technologies minimize cost.

6 steps to the simulation:

1 - The baseline macro-economic drivers, energy supply drivers, and energy service technology characteristics are set for the desired time frame (15 years, 20 years, etc) in five-year increments.

  - Technologies are represented in the model in terms of the quantity of energy service they provide. This could be, for example, vehicle kilometers traveled, tones of paper, or m2 of floor space heated and cooled. Then, a forecast of growth in energy service demand is used to drive future demand for energy services, usually in five-year increments (e.g., 200, 2005, 2010, 2015, etc.).

  - Macro-economic drivers include income, economic output, structural evolution, interest rates, employment rates, regulations, etc. They account for all energy service demands in the energy demand module.

  - Energy supply drivers include international prices, domestic market developments, forecasts of costs and availability of marginal energy supplies, and the resulting forecasts of domestic prices.

  - Energy service technology characteristics include the date a technology or process will be available, technology energy efficiencies and capital / operating costs, firm and household time preferences (discount rates) for discretionary and non-discretionary expenditures (new and retrofit), and other behavioural parameters (differences in intangible costs, perceived risks and consumer surpluses of different technologies, interdependence of certain technology choices that serve different energy services, such as domestic water heating and domestic space heating).

2 - The energy demand module (industry, commercial, residential and transportation) is run for the initial driver values (including energy supply prices).

  - In each future period, a portion of the initial-year’s stock of technologies is retired. Retirement depends only on age. The residual technology stocks in each period are subtracted from the forecast energy service demand and this difference determines the amount of new technology stocks in which to invest.

  - Prospective energy service technologies compete to fill the new service demand. The objective of the model is to simulate this competition so that the outcome approximates what would happen in the real world. Here the model relies heavily on market research of past and prospective firm and household behavior. Technology costs depend on recognized and financial costs, but also on identified differences in non-financial preferences (differences in the quality of lighting from different light bulbs) and risks (one technology is seen as more likely to fail than another). Even the determination of financial costs is not straightforward, as time preferences (discount rates) can differ depending on the decision maker (household vs. firm) and the type of decision (non-discretionary vs. discretionary). As a consequence, the competition is simulated probabilistically; new stock purchases are distributed among prospective technologies with financial costs being just one of several factors in determining market share.

  - In each time period, a similar competition occurs with residual technology stocks to simulate retrofitting (if desirable and likely from the firm or household’s perspective). The same financial and non-financial information is required, except that the capital costs of residual technology stocks are excluded, having been spent earlier when the residual technology stock was originally acquired.

3 - The demand models then send their requirements for energy to the energy supply models. The supply models send back corresponding energy prices. If the changes in energy prices are over a set threshold value for equilibrium (usually 5%), the demand models are then rerun with the new energy prices. This process is repeated until all energy price changes fall under the threshold value. Figure 2 describes this process:



<fig>


4 - Policy or market driven energy price changes may change the lifecycle cost of some energy services, thereby changing the end-use price for these services. If the change in lifecycle cost from the business-as-usual case is beyond a pre-set threshold price elasticities for these services are applied, thereby changing the demand for these services. This new demand is sent back to the supply and demand models for re-calculation.

5 - The model reiterates step 2, 3 and 4 until all variables stabilize. It then proceeds to the next time period. Steps 1 through 5 are repeated for all time periods.

6 - Output is provided for total capital stock, new capital stock, fuel use, service use, emissions and levelized costs by technology for both the business-as-usual and policy scenarios. All price and demand adjustments are also supplied, as well as series of cost outputs related to the demand adjustments. Please see tables 3 & 4.


<hr/>

### Equations/Nodes

Technology Competition and Demand Feedbacks

Operating equations subcomponents (in this order):

  - Technology choice:
    - Demand Assessment
    - Retirement
    - Assessment of capital stock availability
    - Retrofit (ML! get def) and new capital stock competition to meet demand not supplied by existing capital stock

  - Energy supply and demand equilibrium (and goods/services demand)
    - Energy demand models start with exogenous demand for goods/services for the time period (from gov. forecasts).
    - Energy supply models come from exogenous sources or can be calculated internally from the demand model plus net exports.

### Capital Stock Retirement

BSR: Base Stock Retirement (ML! def?):

Starts with base capital stock in a given year, represented as a set of discrete technologies. Retirement of this base stock (Eq. 1) is calculated using a different methodology than capital stock purchased during the simulations (Eq.2) because the vintage of the base stock is a random variable, but new stock vintage is fixed (known).


$$ \text{BSR}_t = \text{max}\large[0, \large( \dfrac{\text{year}_t - 2000}{\text{lifespan}_k}\large)\cdot \text{basestock}_k\large] $$

At time t, for $k^{th}$ tech.



### Prototype

Consider how time is modeled
