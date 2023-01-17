def calc_total_stock_demanded(model, node, year):
    stock_demanded = calc_stock_demanded(model, node, year)
    stock_exported_region_1 = 0
    stock_exported_region_2 = 0

    total_stock_demanded = stock_demanded + stock_exported_region_1 + stock_exported_region_2

    return total_stock_demanded


def calc_stock_demanded(model, node, year):
    sum_service_stock_requested = model.get_param('provided_quantities', node, year).get_total_quantity()
    price_t = model.get_param('price', node, year)
    base_price = model.get_param('price', node, str(model.base_year))
    domestic_elasticity = model.get_param('domestic elasticity', node, year)

    try:
        stock_demanded = sum_service_stock_requested * (price_t/base_price)**domestic_elasticity
    except:
        stock_demanded = 0
        if year != '2000':
            jillian = 1
    return stock_demanded
