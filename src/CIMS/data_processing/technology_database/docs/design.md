# Design

> **Short paths.** `records/` stands for `data/technology_database/`, `output/` for
> `data/processed_data/technology_database/`, and `pipeline/` for the package's
> `pipeline/` folder (`src/technology_database/pipeline/`, or
> `src/CIMS/data_processing/technology_database/pipeline/` in CIMS). The extractions are in
> `data/raw_data/technology_database/` and the documents in `primary-sources/`.

The rules the database follows. The reasoning behind each rule is in the dated
records in `archive/decisions/`, indexed in [decisions.md](decisions.md). This page
states the rules only.

---

## 1. Scope

- **In scope:**
  - industrial technologies that provide an end use in the CIMS hierarchy:
    - heat supply at every temperature and for every energy carrier;
    - the process technologies that transform materials;
    - machine drive, refrigeration, space conditioning and lighting;
    - on-site power generation, including power-only;
    - mobile equipment at industrial sites;
    - CO2 capture and direct air capture;
    - methane venting, flaring and fugitive controls;
    - water and effluent handling.
  - Existing stock and future options both. A legacy type is included while it is
    still in Canadian stock.
  - Prospective options generally at TRL 5–6 or higher.
- **Out of scope:**
  - heat distribution and steam-trap efficiency;
  - downstream fabrication and assembly;
  - residential, commercial and transport end uses;
  - non-GHG pollutants.
- **Scope 1 only.** Purchased electricity is priced by the model, not by the database.

## 2. Archetypes and examples

An **archetype** is a technology option a model chooses between. It provides one or
more end uses. An **example** is one technology as one source presents it, in one of
that source's own contexts (a process or table band). Every example belongs to one
archetype, or is marked `(none)`, deliberately out of scope.

**A new archetype needs one of these:**

- energy use differing by roughly 15% or more from its nearest archetype;
- emissions that differ in kind, or in how they are abated;
- a legacy type that is still in Canadian stock.

Otherwise it is an example of an existing archetype. Size classes, vintages, regions,
vendor designs and product names are never archetypes.

**Carriers.**
- Carriers are a list on the archetype, not a separate archetype.
- The exception is a carrier that changes the equipment itself: a solid-fuel
  boiler, an electrode boiler, a heat pump.
- Differences between carriers on one archetype show up as the range of its values.

**Efficiency.**
- Incremental measures on one device are bucketed into two modifiers:
  `efficiency-moderate` (up to about 15% saving) and `efficiency-strong` (more).
- The measures in a bucket are named in the archetype's `examples-named`.
- An option a model would choose in its own right is an archetype, not a bucket:
  a condensing boiler, a shoe press, a heat-pump dryer, a variable-speed drive.

**Capture.**
- A host that can carry CO2 capture gets its own archetype with a capture modifier:
  `ccs-concentrated` or `ccs-diffuse`.
- That archetype is the host plant with capture (ruling 2026-10-10, superseding
  2026-10-02): it has the host's size (annual output, capacity, capacity factor),
  and its costs, energy use and emissions are the host plant's plus the capture
  add-on's. The add-on comes from same-source evidence first (a source's
  incremental capture cost on the variant, else its plant with capture less its
  plant without: `output/ccs-addon-cost.csv`), else from the median of the
  capture archetypes on the matching CCS node per t CO2 x the CO2 captured per
  unit of the host's product (the variant's `emissions-captured`, else the host's
  CO2 x `share-to-capture` x a 90% capture rate). The parts are kept as targets
  `host-plant` and `ccs-add-on`; `output/ccs-variant.csv` shows both tiers. Its
  emissions are the host's before capture, with `emissions-captured` beside them
  (net = total less captured). Values the variant's own sources give are kept as
  target `as-sourced` (fill rules `var-host-*`, `ccs-host-*`, design s6a).
- In CIMS it differs from its host by a service request to the capture node, so
  the CIMS export writes the `host-plant` part as `fcc` and `fom` plus that
  service request; the capture node's technologies carry the capture cost there
  (ruling 2026-10-10). Energy follows the same split (Brad 2026-11-06, PR06): the
  export writes the host plant's carriers (target `host-plant` and the host's own
  carrier rows), not the variant's, and the capture plant's energy comes through
  the request. The only additions on the CIMS capture technology are the costs and
  energy the capture node's archetypes do not include, listed in
  `records/reference/ccs-addon-extra.csv`, per t CO2 captured: the oxy-fired
  FCC's air separation unit, and the lime kiln's capture heat-supply premium
  (about 105 CAD of capital per t CO2 captured a year). A row with `applies` =
  `tier-2` is a premium the variant's same-source add-on already includes, so the
  fill adds it only to an add-on taken from the capture node; the export always
  adds it. The database variant stays the whole plant, so its
  levelized cost compares with the host's.
- The capture plant itself is a `capture-unit` archetype serving the matching
  `service.ccs.*` node. Its `capture-rate` is the share of the CO2 entering it that
  it captures.
- The plant-level capture rate is `share-to-capture` x `capture-rate` and is not
  recorded (modeller 2026-10-02). A plant-level rate a source gives on a host is
  recorded as its `share-to-capture`; one mapped to a capture plant is excluded
  (`records/reference/plant-level-capture-rate.csv`). A capture rate with no
  species named is of CO2.
- A route that concentrates its own stream keeps its archetype: oxy-fuel kilns,
  top-gas recycling, bath smelting.

**Standalone values.** Every archetype carries absolute values. None is stored as
a delta against a baseline. A negative cost is a saving against a baseline and is
not pooled. An efficiency measure's cost (the cost of adding it to a plant) pools
only as a **whole-plant cost**: the host plant's cost from the same source plus the
measure's. `records/reference/ukerc-host-baseline.csv` names which archetype is
which UKERC host step. A measure whose source gives no host cost is set aside
(`set-aside = needs-host-cost`, listed in `output/set-aside-host-cost.csv`).

**Add-ons and components.**
- Storage added to a base system (a battery on PV, a thermal store on an electric
  heater) is an add-on measure: its cost is `incremental` and pools only as a
  whole-plant cost with the base system's cost from the same source, as above.
- An example has every part its archetype names. A PV system without storage is not
  an example of `photovoltaic-array_battery-storage`.
- Road-vehicle components (engines, batteries, drivetrains) are not proxies for
  mobile equipment.

**Whole-plant archetypes** (a whole mine, refinery or upgrader) are alternatives to
the unit archetypes inside them. A model uses one framing or the other, never both:
where the hierarchy has the unit chain, the whole plant is a `reference` archetype,
validated against the chain (s4a, rule 3).

**Components and assembled whole-plant costs.**
- A `component` archetype (type `component`) is a part priced only to build
  whole-plant costs: `anaerobic-digester`, `separator_biogas-upgrading`. It serves no
  hierarchy node and is never offered to the model; the build refuses a component
  that links to a node. Components are shown in the wide views with `applies-to`
  reading `no connection` (2026-09-30; they were left out before).
- `records/reference/whole-plant-assembly.csv` lists each whole-plant cost to build
  from components: the whole-plant archetype, the source, the parameter and unit,
  and the component rows (example, case, and a multiplier where the denominators
  differ). The build writes the sum as a derived row (`tp-<source-id>-aNNN`) whose
  `derived-from` names every component row.
- Capture plants by difference (2026-10-02). The same file, with `kind` =
  `difference`, gives a capture plant's cost as the same source's plant WITH capture
  less the same plant WITHOUT it; with `kind` = `rebase`, it puts a retrofit cost the
  source gives per unit of host product per t CO2 captured. The multiplier is the
  source's own captured CO2 (nameplate t/yr for capex and fixed O&M, actual t for
  variable O&M); rows are `tp-<source-id>-dNNN` on the capture archetype. Pairs are
  listed case against case by hand, never by archetype; the result covers capture,
  compression and any host changes the source counts. Variable O&M is differenced
  only where the source excludes fuel from it. Where the source prints only the
  emissions with and without capture, avoided CO2 stands for captured CO2 (column
  `assumption` = `avoided-co2-for-captured`, conversion class `assumption`).
- Components are added only within one source, and a different denominator is
  reconciled only with that source's own factor. Components from different sources,
  different currency-years or an unreconciled unit stop the build.
- Where the source also prints the total, the assembled row is a consistency check:
  a duplicate of the printed row, not pooled.
- An example whose source prices the whole plant (digester and engine together) is
  an example of the whole-plant archetype (`anaerobic-digester_chp`), not of a part.

**Reference archetypes.** Type `reference` keeps the numbers for a step the CIMS hierarchy does not resolve (food evaporation). Like a component it serves no hierarchy node, and it is shown in the wide view.

**No connection.** Any archetype that serves no hierarchy node (a component, a reference archetype, or one whose route the hierarchy leaves out, such as coal gasification to SNG) stays in the database; its `applies-to` reads `no connection`. It has no row in `cims-hierarchy_technology.csv`.

## 3. Naming

**The one rule, everywhere:** `-` joins words within a component; `_` separates
components. Folders, files, archetype keys, parameters and branch segments all
follow it.

**Archetype key:** `equipment[_discriminator][_modifier...]`

| Part | Rule | Vocabulary |
| - | - | - |
| equipment | One term, always present. Closed list. | `records/vocabulary/equipment.csv` |
| discriminator | At most one component: the single distinction that makes this a different option, such as a route or operating principle. Each term is listed with a one-line definition. | `records/vocabulary/discriminator.csv` |
| modifier | Zero or more, in the fixed order: `efficiency-moderate`, `efficiency-strong`, `ccs-concentrated`, `ccs-diffuse`, `retrofit`. | `records/vocabulary/modifier.csv` |

Examples: `boiler_fuel-fired`, `kiln_dry-precalciner_efficiency-moderate`,
`blast-furnace_top-gas-recycling`, `steam-reformer_methane_ccs-concentrated`.

**Base and variants vs parallel options (modeller 2026-10-09):** a bare
`equipment` key is a base, and the keys that add a discriminator to it are its
variants (`blast-furnace` and `blast-furnace_top-gas-recycling`). Options that are
parallel, neither a variant of the other, all carry a discriminator
(`solar-concentrator_point-focus` and `solar-concentrator_line-focus`;
`autothermal-reformer_hydrogen` and `autothermal-reformer_ammonia`).

**Never in a key:**
- a carrier, unless it changes the equipment;
- a size class, vintage or region;
- a product name, unless the product is the only real difference between
  parallel options on the same equipment (`autothermal-reformer_hydrogen`,
  `autothermal-reformer_ammonia`) (loosened 2026-10-09);
- a temperature-tier name (`low-temp`, `medium-temp`, `high-temp`). A
  temperature class the technology literature uses for the equipment itself is
  allowed when it is the real difference (`heat-pump_very-high-temperature`,
  above 150 degC) (loosened 2026-10-09).

**`label`** is the human-readable name beside the key ("Heat-pump dryer").

**Extraction names** keep the form the extractor gave them,
`equipment_carrier_variant`, using the terms in
`data/raw_data/technology_database/extraction-vocabulary.csv`. They are the example's name
(`source-technology`) and are never renamed to fit the database.

**Source files** are named `author_year_slug[_source]`. A `_meta` sidecar marker is
stripped before the components are counted.

**Parameters** are one component: `capacity-factor`, `emissions-process`. Currency
and unit values keep `_` as their separator: `2013_GBP`.

## 4. Services: the CIMS hierarchy

**The hierarchy.**
- `records/cims-hierarchy.csv` is the definitive list of services and end uses.
- Replace the file with the latest copy whenever it changes.
- The build stamps the snapshot it used, reports the nodes added and removed since
  the last build, and fails on any record pointing at a path the snapshot lacks.
- Changes the database needs from the hierarchy go in
  [hierarchy-change-requests.md](https://github.com/brad-griffin/technology-database/blob/main/cims/hierarchy-change-requests.md).

**`records/archetype-service.csv`** links each archetype to the hierarchy nodes it
serves, many to many.
- A path is written exactly as the hierarchy writes it
  (`<sector>.iron-and-steel.service.hot-metal`), or as a pattern in which `*`
  stands for one segment (`<sector>.*.service.steam`: every sector's steam node).
- A pattern must match at least one node.

**Families** (`records/family.csv`) group archetypes for display and banding. They
are not service keys.

| Kind | Families |
| - | - |
| Services shared across sectors | `heat-supply`, `power-generation`, `machine-drive`, `space-conditioning`, `lighting`, `mobile-equipment`, `co2-capture` |
| Sector families | One per sector, for integral process equipment |

**Heat.**
- **Tiers** are banded on the temperature the process needs: `low-temp` below
  150 °C, `medium-temp` 150–400 °C, `high-temp` above 400 °C.
- Every archetype has three heat tiers (Brad 2026-11-09), each one or more of the
  bands (`; `-separated) or `not-applicable`:
  - `heat-process-tier`: the heat the process needs, uses, or raises inside itself.
    An electrochemical cell takes the tier of its operating temperature. Archetypes
    with no process heat (shaft work, transport, cooling, lighting, power-only
    generation, pressure or membrane separation, methane controls) and heat-supply
    archetypes, which serve a process rather than being one, take `not-applicable`.
  - `heat-output-tier`: the heat an archetype meant to supply heat delivers. The
    heat-supply bands in the workbook follow it. Every other archetype takes
    `not-applicable`.
  - `heat-waste-tier`: the main stream leaving the archetype that waste-heat
    recovery could use, after the recovery the equipment itself includes (a
    boiler's flue gas after its economizer, a cement kiln's exhaust after the
    preheater); `not-applicable` when there is none.
  The judgements and their basis are in `reviews/heat-tier-assignment_2026-11-09.csv`
  and `reviews/heat-output-and-waste-tier_2026-11-09.csv`.
- Two temperature parameters (Brad 2026-11-09). `heat-process-temp` is the
  temperature the process needs to achieve its end use (kiln sintering, bath,
  reactor, cooking or drying temperature). `heat-output-temp` is the temperature of
  the heat an archetype meant to supply heat delivers (steam, hot water, hot gas,
  thermal oil). Each has `-min` and `-max` forms that widen its low and high.
  Neither records the heating side: a flame, flue gas, heating surface, tube wall,
  or the steam or hot gas supplied to a process is ruled out row by row
  (`reference/excluded-row.csv`). `heat-output-temp` counts only on heat supply,
  boilers, and archetypes with heat or steam among their products
  (`pipeline/normalise.py`, `supplies_heat`).
- `heat-waste-temp` is the temperature of a stream leaving an archetype that
  waste-heat recovery could use: exhaust, flue or stack gas, exhaust air, engine
  coolant, condenser heat. Every stream pools as one value; the example row keeps
  the stream as its target. A turbine's or engine's exhaust is waste heat, not
  output.
- Where a temperature exists, it decides its tier: `heat-process-tier` and
  `heat-waste-tier` take every band the central, low and high values span;
  `heat-output-tier` must cover them and may reach further, because sources often
  print only a supplier's maximum. The build lists any archetype whose recorded tier
  disagrees, for correction in `archetype.csv`. On the workbook's heat sheet a tier
  that equals its temperature's bands takes that temperature's shading; a tier no
  temperature decides is shaded as a guess.
- **Integral heat** is released inside the unit doing the transformation (a kiln, a
  blast furnace, an oven). Its equipment is a sector archetype.
- **Supplied heat** comes from a separable system (a boiler, CHP, a fired heater).
  Its equipment is a `heat-supply` archetype on the `service.steam` or
  `direct-heat` nodes.
- CHP is a heat-supply archetype. Its electricity is an output flow, never a second
  technology.

**Annotations.** `records/hierarchy-annotation.csv` records what the database knows
about the nodes it serves, grouped as the former design end uses were. Each group
has the paths it covers and the following fields:
- what it covers;
- its service unit;
- heat demand temperature;
- steam conditions;
- **draws-on**: the groups whose heat or drive the process takes, written
  `target (duty); target (duty)`. A link is a demand, not a second place the energy
  is counted.

Granularity follows **rule 7**: keep a service at the most general aggregation that
still captures a real difference in energy use or emissions. Do not bundle services
when the energy service differs, when the emissions differ in kind or in how they
are abated, or when a policy instrument treats them separately.

## 4a. Footing at a node: what an archetype counts and what it asks for

Archetypes at one CIMS node compete on the same footing, and each piece of a
route is counted at one node only (node alignment review, Brad 2026-11-05;
`reviews/node-alignment-review_2026-11-05.csv`). Five rules:

1. **Supplied heat is a request.** At a node whose sector has a steam or fired-heater
   (`direct-heat`) service, a unit archetype carries no boiler or heater fuel. Where
   its sources include their own boiler or fired heater, that fuel becomes a `steam`
   (or `heat`, bought from the direct-heat node) carrier at the generator's
   efficiency: request = fuel x efficiency, and the total loses fuel x (1 -
   efficiency). The source rows stay in `example-parameter.csv`, excluded with the
   reason, and the request is a derived row (`tp-<source>-hNNN`, conversion class
   ratio). `records/reference/supplied-heat.csv` lists the archetypes, the fuels
   restated, the efficiency, the share of capital cost the unit keeps, and the heat
   node. The rule holds only where the sector's heat node offers the generation
   options the unit's sources describe (Brad 2026-11-05); a unit whose heat is made
   inside the process (integral heat: a kiln, a blast furnace, downhole
   oxy-combustion) or whose sector has no heat node keeps its fuel.
2. **Bought intermediates are requests.** An intermediate made at another CIMS node
   (reduced iron, coke, sinter, anodes, oxygen, hydrogen, methanol, ethanol, captured
   CO2, grain) is a request per unit of the archetype's product, never left out and
   never an energy carrier. Energy carriers and feedstocks become requests one per
   carrier (s10); everything else is a row of `records/reference/service-request.csv`
   (written to `output/archetype-service-request.csv`); an `at-node` column limits a
   row to one of the archetype's nodes where the quantity differs by node (a
   smelter's acid per t of copper, lead or nickel). Each request has one line in
   the sector note.
3. **One framing per route.** Where a whole plant competes with the chain of unit
   nodes inside it, the whole plant becomes a `reference` archetype (no CIMS node)
   and keeps its values. Where a source splits the plant, its unit-scope parts are
   examples of the unit archetypes. A major component the whole plant has and the
   chain lacks (hydrogen storage, a steam plant, an air separation unit) is added to
   the chain, as an archetype, an add-on or a request; small missing pieces are
   left out. A route that skips or supplies a downstream step itself stays at its
   node with a zero or negative request for that step (a thin-slab caster asks for
   no reheating; a dry-forming paper machine gives back the drying its paper node
   asks for). Where the hierarchy has only a whole-plant node, the whole plant stays
   and the units are reference (the 2026-10-22 rule).
4. **Hydrogen where the sources make it.** The electrolytic and CO2-to-fuel routes
   (ammonia, methanol, CO2 to ethanol and to Fischer-Tropsch fuels) are on the footing
   their sources describe, which is on-site electrolysis: the electrolysers' energy
   and cost are in the archetype, and it asks for no hydrogen. Examples that buy
   hydrogen are restated by adding the same source's (or the central electrolyser's)
   energy and cost. Units whose sector has its own hydrogen node (refining,
   upgrading, steel) buy from it. The CIMS hydrogen sector is to be revised later to
   choose between on-site and pipeline hydrogen (Brad 2026-11-05).
5. **Flows come from the same scope as the parameters.** Route-level, whole-mill and
   baseline-output rows are kept off unit archetypes' flows (they stay on the
   reference whole plant), and flow units are put on one basis before pooling.

**Validation of the chain against the whole plant.** Each reference whole plant
replaced by a chain has a row set in `records/reference/chain-validation.csv`: the
component archetypes, the quantity of each component's product per unit of the
whole plant's product, and the carrier a component makes for another
(`supplies`). The build writes `output/chain-validation.csv`: the chain's energy by
carrier (less what the chain makes for itself), capital cost, fixed operating
cost, direct emissions and calculated levelized cost beside the whole plant's own,
with the ratio. The comparison itself changes no value: a ratio far from 1 points at
a component the chain misses or at a basis difference to look into.

**Capital the chain misses.** Where a chain's capital cost falls well below its
reference plant's (offsites and utilities, site integration, steps with no
archetype), the gap is added back as a capital share on the units that belong only
to that route, never on shared heat supply, hydrogen or haulage (Brad 2026-11-05).
`records/reference/capital-share.csv` gives each unit's multiplier, set once from
the chain validation so the chain then matches its reference plant; every
archetype at the route's nodes takes the same multiplier, so the options there
still compete on the same footing. Sourced capital costs are multiplied row by row
(the source's value stays in the note); judged ones are multiplied in
`fill-judgement.csv`. The multipliers in use: refining and upgrading process units
x 2.39 (fitted to three refineries and the upgrader), oil sands extraction and froth
treatment x 2.51, kraft pulping, bleaching, evaporation, recovery and lime kilns
x 1.63, paper machines x 1.53, and hydrogen direct reduction x 1.5. A multiplier
below 1 takes off the share of a source's capital for parts the archetype now buys
from another node (oxy-fuel kiln x 0.81 and calcium-looping kiln x 0.873, whose air
separation and CO2 purification units are on the oxygen and capture nodes).

## 5. Parameters, units and bases

**Layer 1 keeps values as reported:**
- only values the source prints, or values derived using only values the source
  prints (a sum, ratio or unit conversion), stored as `derived` with the `formula` and
  a note naming the printed inputs. Never a midpoint invented for a printed range, an
  outside factor or an assumption. A whole-plant cost the build assembles from
  component rows is not also summed by hand in the extraction;
- the source's own unit;
- money in `YEAR_CUR` (`2013_GBP`), with `currency` holding the currency-year only;
- no deflation or exchange.

**Basis flags:**
- `cost-basis` (equipment / installed / total-plant / overnight) and
  `capacity-basis` are required wherever `records/vocabulary/parameter.csv` says so.
- A row missing its required basis is marked `review`, with the note
  `BASIS NOT RECORDED`. It is not pooled until the basis is read back out of the
  source.
- 2026-10-02 read-back: capacity bases were set from what the extraction notes
  record of the source (a capacity in tonnes of the archetype's product is
  `product-output`; "thermal output", "steam output", "MWe", "electric input" and
  the like as stated), and for the EPA 2007 and 2017 CHP catalogs from the
  catalogs themselves (systems are rated by electric capacity; the 2007 annual
  outputs equal capacity x 8760 h x the stated capacity factor). The tag added to
  the note ends "basis read back 2026-10-02". A capacity the source does not
  qualify (an engine rating, an "energy" MW) stays in review. Cost bases the
  extractor recorded as `COST BASIS NOT STATED` cannot be read back: the source
  does not say. A capex with no stated cost basis is taken as installed (modeller
  2026-10-02); the note says `COST BASIS ASSUMED: installed`.

**Product unit.**
- Each archetype declares one `product-unit`: `t` for a process archetype, `GJ` for
  an energy service, `m3` where the product is a volume.
- Every example is converted to it. What cannot be converted is excluded, with the
  reason on the row.
- A source's generic `/unit` denominator is read as the service unit that source
  reports for the step (UKERC: t of product, GJ or m3), and as the product unit only
  when the source names none. A service unit that is not a physical product (UKERC's
  per £k of output) is never pooled.
- Incremental costs and hosts (modeller 2026-10-02): host-plant costs are needed
  where a competition holds both the host and the host plus an add-on, so an
  efficiency variant's incremental cost is added to its host's (the fill rule
  `est-host-increment-*`, hosts in `records/fill/host-archetype.csv`: the archetype
  name less its last modifier, else the only conventional archetype with the same
  equipment and product unit). The host's cost is assumed the same with and
  without the add-on. CCS is counted only in the CCS competition: a capture plant's
  (co2-capture family) incremental cost is its own cost and pools as it stands; a
  host plant with capture holds only the host's cost (see Capture). Archetypes with
  no host are open items in `reviews/sanity-checks/sanity-check-list.csv`.
- A cost the source states as the cost of adding an improvement to an existing
  plant (UKERC measures, UK TIMES "improvement technologies" and add-ons via
  Bataille) is incremental: `cost-basis` is `incremental`. UK TIMES rows are marked
  from their notes at extraction; any other example is listed, with its reason, in
  `records/reference/incremental-cost.csv` (with `case` set where only one case of
  the example is incremental: a leading `[case]` in the note, or UKERC's row code). It is set aside as `needs-host-cost`
  unless the same source gives the host plant's cost, in which case a whole-plant
  row is derived.
- A cost per GJ of energy is put per tonne with specific energy consumption (or a
  per-tonne cost per GJ by dividing by it) only when both count the same GJ: the
  same stream (the same fuel, heat or electricity), the same side (input or
  output) and the same scope (the equipment the cost covers, not the whole host
  process). The check is stated per example and the converted value is graded
  `identity`. Otherwise it is not converted: that would mix input and output
  bases (D13). A cost per GJ of a product's own energy content (per MWh of
  methanol) is not this case; it converts with the product's heating value.
  Amended 2026-09-30; no example qualifies yet.
- A number in another unit is converted with exact unit definitions (Btu, lb, short
  ton, barrel; `pipeline/convert.py`), with a factor **the same source gives for the
  same example** on an output basis (an output capacity, an annual output, a capacity
  factor), or with a **product factor** from `records/reference/conversion-factor.csv`
  (amended 2026-09-26). Each product factor is a physical property with its source:
  a heating value (t or Nm3 of hydrogen to GJ), a density (m3 of ethanol to t), an
  energy content (t of LNG to MMBtu), or a product ratio (t of alumina per t of
  aluminium). It applies to the archetypes its `applies-to` names, or to the
  examples `records/reference/example-conversion.csv` names; there `none` marks an
  example reported per a product the archetype does not make, which is not pooled.
  The row names the factor in `conversion`. An energy input standing in for an
  output is never used, and no factor is written inline in code or in a note.
  A conversion may chain at most two factors (kW of hydrogen -> GJ/yr -> t/yr).
- **Every factor has a class** (amended 2026-09-26), and every converted row
  records the weakest class it rests on (`conversion-class`) and the factors
  themselves (`conversion-factors`): `unit` (a definition, or the currency method),
  `same-source` (the same source's own output, capacity, capacity factor or
  efficiency for the example), `property` (a heating value, density or molar
  mass), `ratio` (a typical process ratio), `internal` (a value pooled in this
  database from other examples), `assumption` (a value chosen where no source
  fixes it). From `property` on the factor is **external** to the source. The
  archetype summaries count the examples that rest on an external factor
  (`n-external`) and name the factors (`external-factors`); the wide view lists
  them in one column, `external-conversions`, and the workbook shades those values
  purple (italics until 2026-09-30). `output/conversion-factor-use.csv` says how many pooled rows each
  factor carries.
- **Reference sources are registered apart** from the technology sources, in
  `data/raw_data/technology_database/source-details_reference.csv`: the constants, densities and
  ratios behind the factors, and the deflator and exchange-rate series behind the
  currency conversion (`used_for` says which), are not evidence about a technology.
  `currency-conversion.csv` names the series each currency factor rests on. Each factor's
  `source-id` resolves there (or, for a factor a technology source prints about its
  own product, in the main manifest); an id may not be in both, and a reference
  source that backs no factor fails the build. `output/reference-source-list.csv`
  lists them with the factors each backs.
- **Energy is on a higher heating value (HHV)** (amended 2026-09-26). A number the
  source states on HHV is used as it is; one whose basis the source does not state
  is read as HHV; one stated on LHV is converted with the HHV/LHV ratios of the fuel
  in and the product out that `records/reference/heating-value-stream.csv` names
  for the archetype (`pipeline/heating.py`), and is not pooled on an archetype with
  no row there. `heating-value` records the basis as the source gave it (HHV, LHV
  or `not stated`) on every row whose value depends on it. Every energy factor in
  `conversion-factor.csv` is on HHV.
  Amended 2026-09-30: an LHV specific energy use takes the ratio of the carrier it
  names in `target` (1 for electricity, heat and steam), not the archetype's fuel;
  before, an LHV electricity or steam use was raised by the fuel's ratio. On a
  per-tonne archetype a cost per kW or MWh of the product's own energy, and a
  MWh-per-MWh energy use, are put on HHV (divided by the product's ratio) before
  the product's HHV per tonne puts them per tonne, which returns the source's own
  LHV per tonne exactly. A cost per kW of an INPUT (`capacity-basis` thermal- or
  electric-input) on a per-tonne archetype is never put per tonne with the
  product's heating value.
  Steam given as a mass is put in GJ with `cf-steam-enthalpy` (saturated steam at
  10 bar(a), 2.777 GJ/t, the most common industrial condition; its note records the
  convention and range). Pulp is counted in air-dry tonnes (90% dry), the common
  unit in pulp and paper data; a figure per oven-dry tonne is converted per example
  (`ODt=ODt-pulp` in `example-conversion.csv`).
  Ratios on file beyond the pure fuels (2026-10-02): black liquor from Larson 2006's
  own LHV and HHV; coke-oven and blast-furnace gas derived from the IEAGHG 2013
  Table C-11 compositions (the water each forms x 44.0 kJ/mol).
  Amended 2026-10-02: an LHV energy use per tonne that names its carrier needs no
  `heating-value-stream.csv` row; the carrier's own ratio applies.
  `heating-value-stream.csv` may hold one row per archetype (a second row used to
  replace the first silently; the build now fails on it).
- **The efficiency bridge** (amended 2026-09-26 and 2026-10-02; `pipeline/bridge.py`).
  An example uses its own efficiency, a derived one where it reports none. It
  counts as CHP only if its archetype is named so or it reports a total CHP
  efficiency: a heat-only boiler on a boiler-or-CHP archetype is bridged by its
  own efficiency, not by the archetype's total less electric (which, as two
  medians of different plants, had come out at 0.3% and inflated emission factors
  several hundredfold). A number
  stated per unit of energy input -- an emission factor per GJ of fuel, a cost per
  kW of input capacity (`capacity-basis` thermal-input or electric-input), a
  capacity rated on its input -- is put per GJ of output with the efficiency: the
  example's own where the same source gives one, otherwise the archetype's pooled
  central value (class `internal`). For a CHP archetype the efficiency is the total
  less the electric one, since its product is the heat. A number per tonne of a
  host plant's product with only the plant's fuel use per tonne (UKERC) is bridged
  only for the examples marked `yes` in `records/reference/efficiency-bridge.csv`:
  the per-tonne number must be the equipment's own, not the whole host process's.
  A specific energy use is never bridged; it would only restate the efficiency.
- **CHP costs per kW of electricity** (amended 2026-10-01). On a heat-supply
  archetype whose product is the heat, a cost or capacity rated per kW (or MW)
  of electric output is put per kW of heat with the heat-to-power ratio: the
  example's own (its printed ratio, else its heat over its electric efficiency,
  else its total less its electric efficiency over its electric efficiency;
  class `same-source`), otherwise the median of the archetype's examples' own
  ratios (class `internal`, factor `hpr:<archetype>`). Applies to every source.
- An absolute cost is put per unit of capacity with the same source's capacity for
  the same case (a leading `[case]` in the note, or UKERC's row code); a capacity
  rated on input is then bridged.
- A methane slip (a % of the methane handled that is lost) is put per GJ of methane
  delivered as s/(1-s) divided by the heating value.
- A fixed O&M a source gives as a percentage of capital per year is put on the same
  example's pooled capital cost.
- **Levelized costs** (`cost-levelized`) are shown with the costs. They are pooled
  only with each other, never with capex or opex, and a levelized cost per unit of
  electricity is not pooled on an archetype whose product is heat.
- A derived row that restates another row of the same example in another unit is
  a **duplicate**. It names that row in `duplicate-of` and is not pooled.
- **Set aside until an efficiency is known.** UKERC costs on energy-service
  archetypes are stated per tonne of the host plant's product, with only its fuel
  use per tonne. They carry `set-aside = needs-efficiency` until the example is
  reviewed in `efficiency-bridge.csv`, and `set-aside = not-bridged` (with the
  reason) where the review found the per-tonne numbers to be the host process's.
  UKERC's per-GJ restatements (per GJ of fuel input) are duplicates of the
  per-tonne rows. `output/set-aside-efficiency.csv` lists both.

**Target units** (pu is the product unit):

| Parameter | Unit |
| - | - |
| `capex`, `opex-fixed` | 2025_CAD per (pu/yr of nameplate capacity). A per-kW rating is put per year at 8,760 hours; a cost over a source's annual output is put at nameplate with its capacity factor (amended 2026-10-10: before, a stated capacity factor was applied, mixing nameplate and output bases). |
| `opex-variable`, `opex-total` | 2025_CAD per pu |
| `abatement-cost` | 2025_CAD per t CO2-eq |
| `specific-energy-consumption` | GJ per pu |
| `emissions-*` | t per pu. An emission factor per GJ of fuel is not a factor per GJ of output and is not pooled with one. |
| `capacity` | MW when pu is GJ; kt/yr when pu is t (t/d at 365 days) |
| shares, efficiencies, rates | %. A COP is a % efficiency. |
| `lifetime`, `construction-time` | years |

**Currency:** deflate in the source currency with its GDP deflator, then exchange at
the 2025 rate.

    factor(YYYY_XXX -> 2025_CAD) = deflator[XXX][2025] / deflator[XXX][YYYY] x exchange[CAD_per_XXX][2025]

- Convert once, from the published currency. The factor is not reversible.
- The reference tables come from the CIMS repository and are refreshed there.

**Emissions** are four parameters, kept separate:
- `emissions-combustion`;
- `emissions-process`;
- `emissions-total`, only where a source reports fuel and process together;
- `emissions-captured`.

Species stay explicit in `target` (`co2`, `co2e`, `pfc`) and are not converted to
CO2e. The exception is `emissions-reduction-potential`: it is one percentage,
whether the source counted CO2 alone or CO2-equivalent, so its species are pooled
into one column. A cost per t CO2 (abatement or capture cost) stands for a cost per
t CO2-eq (ruling 2026-10-02).

**Discount rates:** record the source study's own rate and replicate it when
inverting a levelized cost. Apply the model's hurdle rate only afterwards.

## 6. Summarising examples into an archetype

For each archetype and parameter:

| Field | Rule |
| - | - |
| central | The median of the examples' values when three or more examples carry one. Otherwise the value from the example whose source ranks highest in `records/reference/source-priority.csv`. A value reached through a compilation carries the compilation's rank. |
| low, high | The smallest and largest value any example gives, including ranges the sources report themselves. Written only where the values spread. |
| n | The number of examples behind the numbers. Always shown. |

Within one example, an as-reported value is preferred to one derived from it. Several
rows for the same quantity (sizes, cases) count as one example, taking their median.

## 6a. Filling empty cells

The aim is a value in most or all parameters, and from 2026-10-09 a value (or a
deliberate marker) in every cell CIMS reads: capex, fixed and variable O&M, energy
use or efficiency, lifetime, annual output, TRL. Where the examples give none, the
fill step (`pipeline/fill.py`, after the summaries) writes one, graded by how it
was reached. Adopted 2026-09-30; widened 2026-10-09 (the fill-methods proposal,
`reviews/fill-methods-proposal_2026-10-03.md`).

| Grade | What it means | Workbook |
| - | - | - |
| `sourced` | Pooled from examples, as in section 6. | darkest tint (the same with an external factor) |
| `identity` | Calculated from the archetype's own values by a definition or a physical relation: 100 / efficiency, fuel use times its emission factor, combustion plus process emissions, capacity times capacity factor. | second tint |
| `estimated` | Borrowed or scaled from another archetype, or set by a class rule fitted on this database's own values: a host plant's cost plus an increment, a sibling's lifetime, the equipment class's median, a cost ratio. | third tint |
| `proxy` | A class constant from a named outside catalogue or handbook, not extracted as a source: `records/fill/fill-assumption.csv`, each row with its source, registered in `source-details_reference.csv`. | fourth tint |
| `guess` | A judgement with a range, rationale, basis and author, in `records/fill/fill-judgement.csv`. | lightest tint, grey italic |

and one marker that is not a value, `not-applicable`: the parameter does not
characterise the archetype (the energy use of a leak-detection programme, the
variable O&M of a boiler). It is written with its reason, counts as covered, is
shaded grey and reads "not applicable" in the wide view, and the CIMS exporter
writes nothing for it.

Rules:
- A fill is written only where there is no sourced value, and is recomputed on
  every build, so a new source displaces it with no edit. Examples and extractions
  are never changed.
- A fill takes the worst grade among its rule and its inputs. An estimate that
  scales a value of the same archetype (fixed O&M from its capex) may take a
  filled value and inherits its grade; a sibling or class pool holds sourced
  values only, so the back-test is clean.
- Rules run in file order, then the judgements, then the rules once more, so a
  rule that needs a cell a judgement wrote finds it.
- Every rule is a row of `records/fill/fill-rule.csv`: grade, method, scope,
  formula, rationale, and `status` (active or off). A rule found to give wrong
  values is turned off with the reason, not deleted.
- Section 5 still binds: units and product units must match, a cost per GJ goes
  per tonne only under the same-GJ condition, and no regional adjustment.
- A filled row has `n` = 0, `central-from` saying how it was reached, and `inputs`
  naming the cells and factors it used with their grades.
- Every equipment term has a row in `records/fill/equipment-class.csv`: its
  lifetime class (process plant, heat equipment, power equipment, rotating
  equipment, electrochemical, controls and programmes, mobile equipment, wells and
  pipelines) and its variable-O&M class (`material`, `minor`, `none`).
- **The build writes the summaries as they stood before this step** to
  `output/unfilled/` (`archetype-parameter.csv`, the secondary parameters and
  the three wide views), so every check, review and sanity pass can be run on
  the sourced values alone and on the filled set. `output/coverage.csv` gives,
  per archetype, the grade or marker of each CIMS cell.

Judgements (`fill-judgement.csv`), ground rules adopted 2026-10-09:
- one row per archetype, parameter and target; central within low–high, in the
  parameter's target unit; a cost's range at least a factor of two, an energy
  use's at least ±30%, a plant size's at least a factor of three;
- `basis` is one of `analogue-archetype`, `held-source-figure`,
  `engineering-relation`, `experience` or `added-carrier`, so the weakest guesses are findable;
  `added-carrier` is a carrier another example names that the archetype's split leaves out
  (output/carrier-check.csv), added by hand: it adds to the energy-use total rather than
  being scaled to fit inside it (S160, 2026-10-28);
- the rationale names the analogue or relation and the step to the number;
- `author` is `claude-draft` until the modeller accepts; `status` is `draft`,
  `accepted`, `changed` or `rejected` (a rejected row stays, with the reason, so
  it is not redrafted); draft, accepted and changed rows are used;
- a guess on a cell that has since gained a sourced or calculated value is not
  used; a guess replaces a pooled row that has no central and no usable range.

Rules in force (2026-10-09), in the order they run:
- `est-range-midpoint`: a sourced row with a range and no central (the source
  printed a range only) takes the midpoint; the row keeps its sources and n.
- `id-sec-from-efficiency`: energy use per GJ of output = 100 / efficiency, on
  energy-service archetypes other than CHP.
- `id-efficiency-from-sec`: re-enabled with a carrier condition: GJ-product
  archetypes, not CHP, whose all-carrier total agrees with its carrier split,
  and only a result between 10% and 800%.
- `id-single-carrier-split`: an archetype listing one carrier: that carrier is
  the unnamed `fuel` row where there is one, else the total.
- `id-combustion-co2-from-fuel`: named fossil carriers times the ECCC NIR CO2
  factors on HHV (IPCC defaults only where the NIR gives none). Not where CO2
  process emissions are recorded, where only an unnamed `fuel` is known, or on
  reformers, gasifiers and reactors, whose fuel may be a feedstock.
- `id-sec-from-emissions`: the inverse, where a source gives combustion CO2 and
  the archetype burns one fossil carrier.
- `id-zero-combustion-no-fossil`: combustion CO2 = 0 where every carrier is
  non-combustible or biogenic (biogenic CO2 is CIMS's emissions-biomass).
- `id-total-from-parts`: total emissions = combustion + process, same species.
- `id-opex-variable-from-total`: opex-total less opex-fixed.
- `est-host-increment-capex`, `-opex-fixed`; `est-ccs-host-*`: as before
  (section 2 and 5).
- `est-modifier-ratio-capex` (active again): a variant from its base times the
  same-source variant/base ratio; `est-bucket-increment-capex`: where no such
  pair exists, the host cost times one plus the median increment/host ratio the
  database's own set-aside measure costs give for that modifier.
- `est-capex-per-gj-via-sec`: a capex per GJ-yr of heat output on a per-tonne
  archetype goes per t-yr with the archetype's pooled energy use (class
  internal), under the same-GJ condition; graded estimated because the factor is
  pooled, not the example's own (ruling 2026-10-09).
- `est-sibling-lifetime`, `-construction-time`, `-capacity-factor`: nearest
  siblings, any product unit (2026-10-09); `est-class-lifetime`: the equipment
  class's median where no sibling has one; `proxy-class-lifetime`: a catalogue
  life for the class where the class pool is short.
- `id-annual-output-from-capacity`: capacity x capacity factor x hours;
  `est-end-use-annual-output`: the median output of the archetypes on the same
  end use (same product unit; sourced and identity values; one or more);
  `est-sibling-annual-output`: tier-1 siblings only. Plant scale is a convention
  the model installs, not the cost it competes on (costs are per unit of annual
  capacity), so these ranges are x3 either side.
- `est-bucket-saving-sec`: a variant's energy use = its base's x (1 − s), with s
  the variant's sourced emissions-reduction-potential, else the bucket definition
  (fill-assumption `bucket-saving`, proxy); every carrier scaled alike. The UKERC
  stage (7c) does the same for UKERC members before summarise.
- **Efficient variants are set on same-source ratios** (Brad 2026-11-06). A
  variant's energy use is its host's pooled total x (1 − s), with s read from a
  source that prints both levels for the same process (`pair-saving.csv`: the
  improved level against the base level, total against total where the source
  gives totals, else on its main carrier). The variant's absolute levels with no
  same-source base are ruled out (`excluded-row.csv`), and so are its own carrier
  rows: its carriers come from the host's split of the derived total. The exception is a variant whose sources show the step shifting the mix (`variant-own-carriers.csv`; so far the scrap-preheating arc furnace, which trades electricity for burner fuel): it keeps the carrier rows of its own examples that report electricity and a fuel both, scaled to the derived total. A pair
  outside 0–90% is not applied. An archetype whose sources describe a different
  technology rather than a step on the host is typed `alternative` and keeps its
  own absolute pool (14 retyped; `reviews/efficiency-variant-ratio-test_2026-11-06.csv`).
- `est-sibling-efficiency`: GJ-product archetypes take the median efficiency of
  non-CHP siblings (range at least ±15% of the central). Never per-tonne process
  archetypes.
- `est-cost-ratio-opex-fixed`: capex times the median fixed-O&M share of capex of
  archetypes with both sourced (same equipment with 3 or more, else family with
  5, else all, else any product unit); range the pool's quartiles.
  `proxy-om-share-opex-fixed`: the catalogue share for the class where no pool
  reaches the minimum (fill-assumption `om-share`).
- `est-cost-ratio-opex-variable`: the same for variable O&M, on `material`
  classes only. `na-opex-variable-minor`, `-none`: the marker on the other
  classes (variable O&M folded into fixed O&M, or none).
- `na-energy-methane-control`: the marker on venting, fugitive and flaring
  archetypes whose service is t CH4 or m3 of gas (controls that draw power are
  excluded and valued); `na-energy-not-characterising`: on wind, solar,
  geothermal, storage, pipelines, flares, programmes and pneumatic devices.
- `proxy-typical-plant-size`, `proxy-motor-class-efficiency`: off (no cited
  rows yet; nothing to fill).
- **Back-test.** Every borrowing rule is also run on the cells that have a sourced
  value, as if they had none (`output/fill-backtest.csv`; the build prints the
  hit rates). At 2026-10-09: sibling lifetime 168/192 inside, class lifetime
  211/214, sibling capacity factor 70/78, construction time 47/52, sibling
  efficiency 34/46, end-use output 14/20, fixed-O&M ratio 99/133 (35 within
  25%), variable-O&M ratio 32/56 (3 within 25%: a placeholder, as the proposal
  said), modifier ratio 4/5.

**Variants consistent with their hosts** (ruling 2026-10-10). After the rules and
judgements, a post phase (`phase=post` in the rule's scope) makes every variant
with an active host in `host-archetype.csv` take the host's annual output,
capacity and capacity factor (`var-host-*`), and every host-with-capture variant
the host plant's capex, fixed and variable O&M, energy use (every carrier) and
emissions plus the capture add-on (`ccs-host-plus-addon-*`, `ccs-host-emissions`;
section 2). These override what the variant had; a sourced value moves to
target `as-sourced`. `est-ccs-host-capex` and `-opex-*` remain only as a fallback
where no add-on can be derived.

**Calculated levelized cost** (ruling 2026-10-10). A final phase writes
`cost-levelized` with target `calculated` on every archetype: (capex x capital
recovery factor + fixed O&M) / capacity factor + variable O&M, at the
archetype's lifetime and a reference 8% real discount rate (fill-assumption
`as-discount-rate`; the database's sources that state a rate give a median of
8.5%). Energy is excluded. Target `calculated-incl-energy` adds each carrier's
energy use x its reference 2025 price in `records/reference/energy-price.csv`
(kind `sourced` with a registered reference source: electricity from Hydro-Quebec's 2025 large-power comparison, natural gas and oil from CER Canada's Energy Future 2026 industrial end-use prices for 2025; coal, coke, petroleum coke, propane (LPG), naphtha, gasoline and biomass from US EIA series (SEDS, Quarterly Coal Report) deflated and exchanged to 2025 CAD; `derived`: steam and heat at
the fuel-fired boiler's own calculated-incl-energy cost, hydrogen at the methane
reformer's, ammonia and methanol at the steam-reforming ammonia and methanol
plants' cost per t over their HHV per t (column `derived-per`); `internal`:
by-product gases, black liquor, biogas, municipal waste, waste heat, solar and
geothermal at 0; `pending`: no price yet), written only where every carrier has
a price. A levelized cost a source prints stays in the column with no target.
`est-class-capacity-factor` gives every archetype a capacity factor (class
median, else all). Both are in the main wide view.

Still missing after the fill (2026-10-09): fossil combustion CO2 on 45
archetypes whose `carriers` cell in `records/archetype.csv` is blank
(`reviews/carrier-proposals_2026-10-09.csv` proposes the carriers) and on the
reformers, gasifiers and reactors the CO2 identity excludes by design.

## 7. Traceability

Every fact row carries:
- `source-id`, the original the number belongs to;
- `via-source-id`, the compilation it was read from (UKERC, IPCC);
- `source-file` and `source-locator`, which must land a reader on the number;
- `extraction-file`;
- `value-type`: `reported`, `derived`, `assumed`, `proxy` or `placeholder`.

A `derived` row carries its `formula` and the `record-id` it was derived from.

**A compilation's row never stands beside the original it cites.** When the
original is held, its own figure is used and the compilation's row is dropped,
unless the compilation states a reason for its different number (2026-10-03).
A figure the original does not contain at all is the compilation's own and is
credited to the compilation.
For UKERC the row-by-row outcome is `records/reference/ukerc-original-reconciliation.csv`;
for a compilation extracted afresh (N-ZIP over IFS) the extractor leaves the
reprinted rows out of the main file and lists them in `_other-values`.

**Record ids** are `tp-`, `tf-` or `tc-` followed by the source-id and a four-digit
number, numbered within the source (`tp-ukerc_2013_001-0042`), so an edit to one
source never renumbers another.
Rows the build derives carry a letter and three digits instead: `-wNNN` (UKERC
whole-plant), `-aNNN` (assembled), `-dNNN` (capture cost by difference or rebased),
`-cNNN` (capture energy restated per t CO2), `-xNNN` (a source correction, below).

**Figures a source prints wrongly** (2026-10-08). Where a source prints a figure
that cannot be right as it stands -- a slipped decimal point or zero, a wrong unit
or carrier label, a digit too many, a row shifted against its label -- and the
source itself (its totals, its other tables, the sibling rows, physical bounds)
shows what was meant, the corrected figure is taken into the database rather than
left out. The extraction keeps the figure as printed; the correction is made once,
in `records/reference/source-correction.csv`, with the figure as printed and the
evidence, and marked on every row it touches: the `correction` column holds the
correction-id, the note starts `ASSUMED CORRECTION (sc-NNN)`, and the row's
conversion-class is `assumption` (factor `assume:source-correction-sc-NNN`).
The summaries count such examples in `n-corrected` (and in `n-external`), and the
wide view lists them in `source-corrections`. A replaced row stays in
example-parameter.csv, excluded as superseded. Rows the extraction had already
corrected before this rule are marked the same way. Where the evidence does not
settle what was meant (two readings both plausible), the figure stays out and the
reason stays in the extraction's notes.

**Registration.** Every source is registered in `data/raw_data/technology_database/source-details_sources.csv`
and `source-details_files.csv`; a reference source (a constant, a conversion
factor's source, a currency series) in `source-details_reference.csv` instead. The build takes source ids from there. The
manifest records what the database draws on: sources that provide data, the
originals a compilation cites (registered but not held), and sources queued for
extraction.

**Documents to get.** Documents worth obtaining are listed in
`source-intake/to-get.csv`, with priority, the source that cites each one, and
what it would supply. A document leaves the list when it is downloaded and
registered. Documents that turn out to be unobtainable move to `source-intake/unavailable.csv`, with the reason and any substitute held, so they are not searched for again.

**Background documents.** A downloaded document that yields nothing for the
database goes to `source-intake/background-no-data-extracted/ (its `_meta.csv`; the document in `primary-sources/background-no-data-extracted/`)` and is recorded
in `data/raw_data/technology_database/background-register.csv` (why it has no data, what little it
has), not in `source-details_files.csv`. Its source row stays only if a
compilation cites it as an original.

## 8. Readiness

**One ladder.**
- `trl` is on the common 1–9 ladder of
  `cims/service-technology-map/technology-readiness-levels.md`, using its
  industrial-process reading.
- Other scales (IPCC's NASA TRL, the IEA ETP scale) map through
  `records/reference/trl-scale-map.csv`.
- A reported range is carried at its lower bound.

**Availability rule, for the CIMS `available` parameter:**

| TRL | Available |
| - | - |
| 9 | now |
| 7–8 | in 5 years |
| 5–6 | in 10 years |
| ≤ 4 | in 15 years, or unavailable |

## 9. Reading the outputs

- The workbook's archetypes sheet has one row per archetype, including blank ones and
  those that serve no hierarchy node (`applies-to` reads `no connection`). `end-uses`
  names the annotation groups each archetype serves. `core-sourced` (also in
  `archetype.csv`) counts how many of the seven core model values (capex, fixed
  opex, variable opex, energy, lifetime, annual output, emissions; `fill.CIMS_CELLS`)
  come from a source, 0 to 7 (Brad 2026-11-09; it replaces the `status` flag, which
  marked an archetype quantified on any one sourced value).
- The heat sheet holds the three heat tiers and their temperatures, process, output,
  then waste, for the same rows as the archetypes sheet.
- `filled` lists the parameters the fill step wrote and their grade or marker; in
  the workbook every value is shaded on one ember scale by how certain it is
  (sourced and converted on the darkest tint, identity, estimated, proxy, then
  guess in grey italic on the lightest tint; not-applicable unshaded light grey
  italic; `pipeline/workbook.py` SHADE, key on the readme sheet), and filled
  values' `_n` is 0. `output/unfilled/` holds the same
  tables without any fill; `output/coverage.csv` the grade of each CIMS cell per
  archetype.
- The wide view is built for the workbook only; the wide CSVs were dropped
  (Brad 2026-11-06). It has `wide-archetype` (the archetypes sheet, one column set
  per parameter) and `wide-carriers-species` (specific energy by carrier, emissions
  by species, alphabetical; split into the energy, emissions and feedstocks sheets,
  2026-11-04). The secondary parameters (values by stream or part, load-range-min,
  degradation-rate and a non-CHP efficiency-chp) are in long form only.
  Where each (parameter, target) goes, and what is dropped from the wide views, is
  `pipeline/placement.py`. Secondary parameters are in `archetype-secondary-parameter.csv`,
  not `archetype-parameter.csv`. Targets that restate the base are pooled with it first
  (`pipeline/roll_up.py`); the CHP efficiency (total CHP efficiency on CHP archetypes),
  the temperature ranges (heat-output-temp and heat-process-temp -min/-max) and the
  single-stream efficiency
  (repeated in efficiency_heat or efficiency_electricity) are roll-ins on the
  summaries (`pipeline/roll_in.py`), so both CSVs carry them. In the workbook each
  parameter's `_low`, `_high` and `_n` columns are folded into an outline group beside
  the central column, with their own header shading.
- `product-density` and `product-energy-content` give the conversions for products
  stated in more than one unit (`records/reference/product-property.csv`).
  `secondary-products` names what a multi-product archetype also makes; an archetype is
  classified by its primary product (CIMS negative service requests for the others).
- Units are the final product of the end use, except for services that treat a stream,
  where the stream handled is the service.
- A source that does not state its heating-value basis takes its publisher region's
  convention (`records/reference/source-region.csv`) unless a ruling in
  `source-heating-basis.csv` names it.
- Families are banded cross-cutting services first, then sectors alphabetically.
  Within each family the workbook bands by end use in process-chain order
  (`pipeline/bands.py`); an archetype serving several end uses repeats under each
  (workbook only, styled like any row), in the family whose sector owns that end use;
  the `families` column lists them, home family first. Machine drive stays banded by equipment.
- **Never sort, average or chart a cost or intensity column across families.** It
  holds CAD per tonne on a kiln row and CAD per GJ on a boiler row. Read it against
  `product-unit`.
- The long tables (`example-parameter.csv` and the other `example-*` files) are the
  record of every number. The wide views and the workbook are views, and anything
  typed into them is lost on the next build.

## 10. The CIMS YAML layer

| Database | CIMS YAML |
| - | - |
| `capex` (with `cost-basis`); on a host-with-capture variant its `host-plant` part, plus any `ccs-addon-extra.csv` cost | `fcc` |
| `opex-fixed` (+ `opex-variable` x annual output); on a host-with-capture variant the `host-plant` parts | `fom`, `vom` |
| a host-with-capture variant's CO2 to capture | `service-request` to the CCS node |
| `lifetime` | `lifetime` |
| `annual-output` | `output` |
| `efficiency`, or input flows (on a host-with-capture variant, the host plant's, plus any `ccs-addon-extra.csv` energy) | `service-request`, one per carrier |
| `feedstock-consumption` by carrier | `service-request`, one per carrier |
| `records/reference/service-request.csv` (`output/archetype-service-request.csv`) | `service-request` to the named node, including zero and negative requests |
| `emissions-*` | `emissions`, `emissions-biomass`, `emissions-removal` |
| `trl` → rule above | `available` |
| constraint shares | `market-share-new-max`, `market-share-new-min`, `market-share-total` |

Field names for the YAML files are in `cims/service-technology-map/README.md`.
