from CIMS.utils.parameter import list as PARAM


def prev_stock_existed(model, node, year):
    for year in [y for y in model.years if y < year]:
        pq, src = model.get_param(PARAM.provided_quantities, node, year, return_source=True)
        if pq.sum_provided_by_total() > 0:
            return True
    return False