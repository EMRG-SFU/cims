# CCS: linking a capturing host to the CCS node

Status: proposed design, revised 2026-10-22. Implement **after** the current uncommitted
changes on `staging` are committed: indirect removal through intermediates
(`emissions.py`), the flow ledger (`ledger.py`), and the by-product handling of negative
requests. This design reuses the indirect-removal traversal and uses the ledger to check
results.

Aims: keep the standard CIMS technology inputs, add as few special code rules as possible,
and let a capture-rate sensitivity move capture energy, transport, and removal while the
capture plant's capital stays fixed.

## 1. The problem today

A host technology with CCS (a kiln, boiler, or reformer "with CCS") has two links to the
sector's CCS node:

- a fixed `service_request` to the CCS node (t CO2 per unit of product), which sets only the
  service cost the host pays; and
- the CCS technology's `emissions_removal`, which the host inherits and applies to **all** of
  its CO2, combustion and process, direct and (with the staging change) indirect, whatever
  quantity it requests. The comment block in `calc_competition_emissions_cost` states this
  on purpose.

So a plant that captures one stream (an ammonia plant's stripper vent, a blast furnace's
stove flue gas) gets credit for removing all its CO2, and the quantity it pays for has no
link to the quantity removed.

## 2. Design

### 2.1 The CCS node's unit: t CO2 sent to capture

The CCS node provides **t CO2 sent to capture**, meaning the CO2 in the gas the capture
plant treats. The host's request to the node is in that unit.

| Where | Parameter | Unit | Notes |
| - | - | - | - |
| Host tech | `service_request` to the CCS node | t CO2 sent per unit of product | fixed input, as today. From `tech_data`: gross CO2 x `share-to-capture`. |
| Host tech | `fcc`, `fom` | as today | the host plant only, without the capture plant |
| CCS tech | `fcc`, `fom`, `output` | output in t CO2 sent per year | standard inputs. Capital and fixed O&M per tonne sent, so they do not change with the capture rate. |
| CCS tech | `service_request` to energy, VOM, and transport and storage | per t CO2 sent | standard inputs: each per-tonne-captured value x the capture rate |
| CCS tech | `emissions_capture_rate` (renamed from `emissions_removal`) | share, by GHG and emission type | share of the CO2 sent that the plant captures |

These are all standard technology inputs. The CCS node needs no special cost code.

### 2.2 Removal: the one rule change

For host tech *h* in year *y*, over every CCS node *n* it requests, every GHG *g*, and
every emission type *t*:

```
sent[h,n]        = service_request[h -> n]                    (t CO2 per unit of product)
rate[n,g,t]      = market-share-weighted emissions_capture_rate of the techs at n
covered[h,n,g]   = sum of gross[h,g,t] over the types t where rate[n,g,t] > 0
want[h,n,g,t]    = sent[h,n] x rate[n,g,t] x gross[h,g,t] / covered[h,n,g]
removal[h,g,t]   = min(sum over n of want[h,n,g,t], gross[h,g,t])
```

- **gross** is the host's gross emissions per unit of product: direct, plus the residuals of
  supply it reaches through non-supply intermediates (the staging `_node_residuals`
  traversal), fossil and biogenic together.
- **Which types a node captures** comes from the existing keying of `emissions_capture_rate`
  (Context = GHG, Sub_Context = emission type). A concentrated node that treats only a
  process vent carries a rate for CO2 Process and none for Combustion, so its removal lands
  on Process CO2 only. A node with rates on both types spreads its removal across them in
  proportion to the host's gross.
- **One cap per host, GHG, and type, summed over all its CCS nodes.** A host requesting both
  a concentrated and a diffuse node cannot remove more of a type than it emits, however the
  two requests are set. Where the cap binds, each node's share is scaled down in proportion
  (for reporting and the ledger); the node's request, and so its cost, is not changed.
- **Fossil and biogenic**: within each GHG and type, removal is split in proportion to the
  host's fossil and biogenic gross. The fossil part is avoided emissions; the biogenic part is
  negative emissions, as today.

This replaces "the child's removal rate applied to all of the requester's emissions" in
`calc_competition_emissions_cost`, `calc_financial_emissions_cost`, and
`_apply_indirect_removal`. The service request already in the model drives removal; no new
host parameter is needed.

The CCS node's own fuel CO2 (from the gas burned to raise regeneration steam) keeps its
current treatment: the node captures it at its own `emissions_capture_rate`, and the
captured amount is in its transport-and-storage request.

### 2.3 What a capture-rate change does

| Quantity | Changes with the capture rate? |
| - | - |
| Capture plant capital and fixed O&M | no (per tonne sent) |
| Capture energy, variable O&M, and transport and storage | yes, through the CCS tech's requests per tonne sent |
| Removal booked on the host | yes, sent x rate |
| Cost per tonne captured | falls as the rate rises |

This matches NETL's 90% and 99% cases (Hughes and Zoelle 2023): capital rises 3-9% for 10%
more CO2 captured, and the cost per tonne captured is about flat or falls slightly.

**For a sensitivity run, change the rate and the three rate-scaled requests together.** The
data pipeline (`data_processing_calibration`) should hold the capture rate as one input and
write `emissions_capture_rate` and the energy, VOM, and transport-and-storage requests from
it. Changing `emissions_capture_rate` alone in a scenario file moves removal but leaves
capture energy and transport at the reference rate.

### 2.4 Concentrated and diffuse streams

Each sector's single CCS node today is costed on one NETL case (compression only for
ammonia, ethanol, and gas processing; solvent capture for SMR, cement, and steel) but
serves both kinds of host, so a gas boiler with CCS in Chemical Products buys
compression-only capture at a 100% rate. Split each sector's node into `ccs.concentrated`
and `ccs.diffuse`, as the redesigned hierarchy already does, and point each host's request
at the node for its stream. A host with both streams (an ammonia plant's vent and its
reformer flue gas) requests both nodes. Each node's removal follows its own request and
lands on the emission types its rate covers; the cap in section 2.2 applies to the sum, so
the two cannot together remove more than the host emits.

To keep the types right, give each node's `emissions_capture_rate` only the types its
stream contains: the concentrated node CO2 Process (vents, shifted syngas, fermentation),
the diffuse node CO2 Combustion and Process (flue gas that carries both, such as a cement
kiln's). Emission types matter for reporting and for any tax or regulation that treats
types differently; totals are right either way.

## 3. Known limitation: carbon-free fuel substitutes

The host's request is fixed per unit of product. If the CO2 in the treated gas falls
because the host's fuel blend takes on a **carbon-free** substitute (hydrogen in a methane
blend, or partial electrification within the same technology), then:

- the cap keeps removal at the CO2 actually there, so emissions stay right;
- the capture plant's capital and fixed O&M stay, which is realistic: the plant was built
  for that gas flow;
- capture energy, variable O&M, and transport and storage stay at the full request, which
  overstates them; and
- new stock of that technology carries a capture plant sized for the original CO2.

There is **no problem with renewable natural gas**: it produces the same CO2 per GJ as
fossil gas (`emissions_biomass` CO2 0.0508 t/GJ, equal to natural gas's fossil factor), so
the request still matches what the plant captures. The captured tonnes move from avoided
fossil emissions to negative emissions.

Most industrial hosts are separate natural gas, oil, and coal technologies with a fixed CO2
per unit, so the limitation applies only to blend-fed hosts taking on hydrogen. If that
becomes important, add an optional host parameter `emissions_to_ccs` (share of gross
emissions, by GHG and emission type, sent to a named CCS node) that computes the request
each year from actual gross emissions. That would be the one special rule; it is not part of
this change.

## 4. Implementation steps

1. **Rename** `emissions_removal` to `emissions_capture_rate`:
   - `src/CIMS/utils/parameter/list.py`; keep `emissions_removal` as a deprecated alias that
     the readers translate, with a warning, for one release;
   - `src/CIMS/emissions.py` (16 uses) and `src/CIMS/utils/parameter/query.py` (4 uses);
   - `data/model_inputs/defaults/defaults_Parameters.csv`: the `emissions_removal` row
     (default 0, "GHG emissions removal effectiveness"); reword as "share of the CO2 sent to
     capture that the plant captures";
   - the model inputs: change the `data_processing_calibration` pipeline that generates them,
     not the generated CSVs (about 256 rows on the sector and electricity CCS nodes).
2. **Removal rule** (section 2.2) in `calc_competition_emissions_cost`,
   `calc_financial_emissions_cost`, and `_apply_indirect_removal`; replace the comment that
   says removal is not scaled by the request.
3. **Validator** (`model_validation`): warn where a tech requests a node whose techs carry
   `emissions_capture_rate` and the request's unit is not t CO2; warn where the cap binds in
   the base year (the summed requests x rates exceed the host's gross of a type, so the
   inputs are probably wrong).
4. **Ledger**: record captured CO2 as flows (host to CCS node, CCS node to transport and
   storage), and check that captured is no more than gross for every host.
5. **Data** (`data_processing_calibration`):
   - restate the CCS nodes per t CO2 sent: `output` as t sent per year (captured / rate) and
     energy, VOM, and transport-and-storage requests as per-captured x rate;
   - set host requests to t CO2 sent (from `tech_data`: gross x `share-to-capture`);
   - fix the defects in `tech_data/docs/cims-to-fix.md` item 4 and the CIMS comparison report:
     100% removal on the Chemical Products, Ethanol, and Biodiesel nodes; the Petroleum Crude
     boiler requests that ramp to the fuel request; and the Petroleum Refining "NG CCS"
     request in `$`.
6. **Tests**:
   - request 1.0 t sent and rate 0.9 remove 0.9 t, and total capture capital equals
     `fcc` / `output` x 1.0;
   - raising the rate from 0.9 to 0.99 raises removal, energy, and transport by 10% and
     leaves capital the same;
   - an ammonia host requesting the concentrated node for its vent removes only the vent CO2
     and keeps its flue-gas CO2;
   - a methane-blend host at 50% RNG removes the same tonnes, half booked as negative;
   - a host whose request x rate exceeds its gross is capped at gross;
   - a host requesting a concentrated node (rate on Process only) and a diffuse node (rate on
     Combustion and Process) with requests that together exceed its gross: removal is capped
     per type at gross, the concentrated node's removal is all Process, and both nodes are
     still paid in full;
   - a by-product (negative) request does not enter gross.

## 5. What `tech_data` supplies

| tech_data | CIMS |
| - | - |
| host-with-capture variant, `host-plant` cost | host `fcc`, `fom` |
| host gross CO2 x `share-to-capture` | host `service_request` to the CCS node (t sent) |
| `capture-unit` archetype `capture-rate` | CCS tech `emissions_capture_rate` |
| `capture-unit` capex and fixed O&M per t captured x capture rate | CCS tech `fcc`, `fom` per t sent |
| `capture-unit` energy and VOM per t captured x capture rate | CCS tech energy and VOM requests per t sent |

## 6. Open questions

- Declining capital costs (DCC) on CCS techs: capacity becomes t CO2 sent, not captured;
  check the DCC capacity inputs.
- Retrofit (`retrofit_existing_max`) of capture onto existing hosts: unchanged by this design,
  but worth a test.
- LCCs change against the Reference run (removal now follows the request); compare
  market shares before and after.

## Related

- `tech_data/docs/cims-to-fix.md`, item 4.
- `tech_data/compare/cims-model-inputs/comparison/cims-comparison-2026-10-14.md`, section CCS.
- `tech_data/docs/reporting-accounting/reporting-accounting.md`, sections 4 and 6 (the flow
  ledger).
