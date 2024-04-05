import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_prices_change_over_time(model, out_file, show=False):
    """Creates a visualization of fuel prices over time as a multi-line
    graph.
    
    Parameters
    ----------
    model : CIMS.Model
        The model containing data for visualization.

    out_file : str
        Filepath to the location where the visualization will be saved.

    show : bool, optional
        If True, displays the generated figure, by default False
    """

    price_data = get_fuel_prices(model)
    # Set the style of the visualization
    sns.set_theme(style="darkgrid")

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=price_data, x='year', y='price', hue='fuel_type', marker='o')

    plt.title('Fuel Prices Over Years')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend(title='Fuel Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Display Plot
    if show:
        plt.show()

    # Save plot
    plt.savefig(out_file)


def visualize_price_comparison_with_benchmark(model, benchmark_file, out_file, 
                                              show=False):           
    """Creates a visualization comparing prices with their benchmark values.
    A wrapper for the visualize.visualize_price_comparison_with_benchmark()
    function.

    Parameters
    ----------
    model : CIMS.Model
        The model containing data for visualization.

    benchmark_file : str
        The location of the CSV file containing benchmark values for each
        fuel.

    out_file : str
        Filepath to the location where the visualization will be saved.

    show : bool, optional
        If True, displays the generated figure, by default False
    """
    # Example of preparing data for heatmap (you'd need to calculate the differences first)
    # Assuming `data_diff` is a DataFrame with differences calculated
    data = get_fuel_prices(model)
    benchmark = pd.read_csv(benchmark_file)

    # Filter default data to only include years found
    data['year'] = data['year'].astype(int)
    benchmark['year'] = benchmark['year'].astype(int)
    benchmark = benchmark[benchmark['year'].isin(data['year'].unique())]
    data['Price Difference'] = data['price'] - benchmark['price']

    # Get unique years and assign a unique color to each
    unique_years = data['year'].unique()
   
    # Format dataframe for multi-scatter plot
    pd.melt(data, id_vars=['fuel_type','year'], value_vars=['Price Difference'])
    
    # Create plot
    _, ax = plt.subplots()
    for year in unique_years:
        ax.scatter(data[data['year']==year]['Price Difference'],data[data['year']==year]['fuel_type'],label=year)
    ax.legend()

    # Placing the legend outside the plot to the right
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    
    # Adjust the subplot parameters to give some more space for the legend
    plt.subplots_adjust(right=0.75)
    plt.title('Price Difference Heatmap (Run values - Test values)')
    plt.xticks(rotation='vertical')
    plt.tight_layout()

    # Display Plot
    if show:
        plt.show()

    # Save plot
    plt.savefig(out_file)


def get_fuel_prices(model, out_file=None):
    """Retrieves model fuel prices for each year, returning a dataframe meant
     for visualization

    Parameters
    ----------
    model : CIMS.Model
        The model to retrieve data from.

    out_file : str, optional
       Writes a CSV of fuel prices to the provided filepath. None by default, 
       which won't write a file. Meant for recording benchmark data.

    Returns
    -------
    pd.DataFrame
        fuel prices by fuel_type and year
    """    
    fuel_prices ={
        "fuel_type":[] ,
        "price": [],
        "year": []
        }
    for year in model.years:
        for supply_node in model.supply_nodes:
            fuel_prices["fuel_type"].append(supply_node)
            fuel_prices["year"].append(year)
            fuel_prices["price"].append(model.get_param('price', supply_node, year))
    fuel_prices = pd.DataFrame(fuel_prices)

    if out_file is not None:
        fuel_prices.to_csv(out_file, index=False)

    return fuel_prices