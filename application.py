### Exploratory Data Analysis and Visualization
# Importing Modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import geopandas as gpd

# Loading Country GeoData
# Load the Natural Earth dataset
world = gpd.read_file('./data/natural_earth_geo_admin0/ne_10m_admin_0_countries.shp')

# Group ISO A3 codes by continent
iso_a3_by_continent = world.groupby('CONTINENT')['ISO_A3'].apply(list).to_dict()

# Iterate over the dictionary and remove '-99' values
cleaned_iso = {continent: [country for country in countries if country != '-99'] for continent, countries in iso_a3_by_continent.items()}

## Loading of the datasets
## Agriculture & Rural Development 
# Rural population (% of total population) For the 70 percent of the world's poor who live in rural areas, agriculture is the main source of income and employment. 
# But depletion and degradation of land and water pose serious challenges to producing enough food and other agricultural products to sustain livelihoods here and meet the needs of urban populations. 
# Data presented here include measures of agricultural inputs, outputs, and productivity compiled by the UN's Food and Agriculture Organization.
# Dataset from WorldBank (26-May-2023)
df = pd.read_csv("./data/worldbank_agri/API_1_DS2_en_csv_v2_5455649.csv", skiprows=3)
# https://api.worldbank.org/v2/en/topic/1?downloadformat=csv

# Databank dataset inspection
df.drop(columns=["Unnamed: 67"], inplace=True) #Obsolete column with Null values
# df.head()

# Dimensions of DataFrame
# df.shape

# Dataframe Info
# df.info()

# Describe DataFrame
# df.describe()

# Creating a reusable function for plotting missing values
def missing_plot(
        df_missing,
        len_original,
        title='',
        xlabel='Columns',
        ylabel='% of Missing Values',
        cbar_label='% Missing',
        cm=plt.cm.coolwarm,
        hline=False,
        hline_y=0,
        hline_label='Threshold at',
        hline_color='red'
    ):
    # Normalize the missing values to map them to colors
    normalized_values = (df_missing - df_missing.min()) / (df_missing.max() - df_missing.min())
    
    # Create a colormap (you can choose any colormap you like)
    cmap = cm
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(24, 6))
    
    # Create a bar plot with custom colors
    bars = sns.barplot(ax=ax, x=df_missing.index, y=df_missing.values / len_original, palette=cmap(normalized_values))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    
    # Create a color bar legend for reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=df_missing.min()/len_original, vmax=df_missing.max()/len_original))
    sm.set_array([])
    
    # Add the colorbar to the same axis (ax) where the bars are plotted
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)

    if hline:
        # Add a custom horizontal line at y = 2 (you can adjust the y-value as needed)
        plt.axhline(y=hline_y, color=hline_color, linestyle='--', label=hline_label)
        plt.legend()
        
    plt.show()

# Lets have a look at the complete dataset with all the indicators included, what we see is that 2022 have relatively many missing values compared to the majority of the dataset.
# In literature having less than 5% missing values is acceptable but for sake of demonstration we will show how they are dropped. Despite the low amount of missing values there could be some change that we want to leave out.

# Calculate the number of missing values per column
missing_values = df.isnull().sum()
# missing_plot(missing_values, df.shape[0], '% Missing values per Column', hline_y=1, hline=True, hline_label='Threshold at 1%')

# Here is visualized how many missing values per indicator summed over all columns exist. This also provides us with insights on how many missing values we deem acceptable.
# For the visualization we want to use the indicators with minimal missing values, for reference a hline of 2000 is also plotted.
#
# The indicator that will be used for the GeoData Visualization is:
# SP.RUR.TOTL.ZS

group_by_indicator = df.groupby('Indicator Code')
missing_values_per_indicator = (group_by_indicator.apply(lambda x: x.isna().sum()))
missing_values_per_indicator_t = missing_values_per_indicator.transpose().sum()
missing_values_per_indicator_t_sorted = missing_values_per_indicator_t.sort_values()
# missing_plot(missing_values_per_indicator_t_sorted,df.shape[0],'Missing values grouped by Indicator',xlabel='Indicator Codes',hline=True,hline_y=1,hline_label='Threshold at 1%')

# Cleaning up the dataframe for obsolete columns after selection has been performed.
indicator_keep = 'SP.RUR.TOTL.ZS'
indicator_codes_unique = df['Indicator Code'].unique()
drop_indicators = [x for x in indicator_codes_unique if x not in indicator_keep]

# Use boolean indexing to select rows to drop and then use drop method
df.drop(df[df['Indicator Code'].isin(drop_indicators)].index, inplace=True)

# Check if the updated dataframe only contains the desired indicators
df['Indicator Code'].unique()

# Calculate the number of missing values per column
missing_values = df.isnull().sum()
# missing_plot(missing_values,df.shape[0],'Missing values per Column',hline_y=0.5,hline=True,hline_label='Threshold at 0.5%')

# Dropping the year column that was mentioned earlier with too many missing values which is 2022
df.drop(columns=['2022'],inplace=True)

# Reindexing dataframe
df.reset_index(inplace=True,drop=True)

# The cleaned dataset after selecting appropiate indicators
# df

# Calculate the number of missing values per column - which has been reduced to below o.5% per column on average
# print('Current shape of dataframe',df.shape)
missing_values = df.isnull().sum()
# missing_plot(missing_values,df.shape[0],'Missing value % per Column',ylabel='% of Missing Values',hline_y=0.5,hline=False,hline_label='Threshold at 0.5%')

# Remaining missing value data e.g. for 1961
is_missing = df[df['1960'].isna()].index
# df[df['1960'].isna()].head(15)

# Replace missing values with 0, in visualization if the value is exactly 0 we can show it as missing.
df.fillna(0,inplace=True)
# Show updated dataframe for the example of 1960 indexes
# df.iloc[is_missing].head(15)

# Selecting all year column indexes
years = df.columns[4:]
# years

# For visualization purposes we will add column that contains the continent of each country based on the ISO_A3 code.
continents = []

for ISO_A3 in df['Country Code']:
    found = False  # Flag to check if a match is found
    for continent, codes in cleaned_iso.items():
        add = None
        if ISO_A3 in codes:
            if continent == 'Seven seas (open ocean)':
                add = 'Seven seas'
            else:
                add = continent
            continents.append(add)
            found = True  # Set the flag to True when a match is found
            break  # Exit the loop when a match is found
    if not found:
        continents.append('Unclassified')

df.insert(2, 'Continent', continents)

# Continent has been added to the dataframe at column index 2
continents = sorted(df['Continent'].unique())
# continents

# The 53 Unclassified countries which ISO_A3 code was not found have to be removed or updated to belong to the correct continent
# Many here are aggregated values, we will manually select countries that will be updated.
unclassified_codes = df[df['Continent'] == 'Unclassified']['Country Code']

# France FRA Europe
# Norway NOR Europe
# Channel Islands CHI Europe
# Kosovo XKX Europe

# print(df[df['Continent'] == 'Unclassified']['Country Name'].head(10))

keep_unclassified  = ['FRA','NOR','CHI', 'XKX']

# Update the continent for the specified country codes to 'Europe'
df.loc[df['Country Code'].isin(keep_unclassified), 'Continent'] = 'Europe'

# Proceed to drop all rows from unclassified which do not include 'FRA','NOR','CHI', 'XKX'
drop_unclassified_filtered = df['Country Code'].isin([code for code in unclassified_codes if code not in keep_unclassified])
# drop_unclassified_filtered
df = df.drop(df[drop_unclassified_filtered].index)

# Run all three cells below and go to the localhost url where the dashboard is hosted.

# Import necessary modules for creating the dashboard
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
# from IPython.display import display, HTML

# Assuming you have a list of years defined somewhere in your code
# Extract the minimum and maximum years from the list
years = [int(year) for year in years]  # Convert the list of years to integers
min_year = min(years)  # Find the minimum year
max_year = max(years)  # Find the maximum year

# Create a Dash web application
app = dash.Dash(__name__)

# Assuming you have a DataFrame 'df' and you want to filter it
# to only include rows where 'Indicator Code' is 'SP.RUR.TOTL.ZS'
df_rural = df[df['Indicator Code'] == 'SP.RUR.TOTL.ZS']

# Calculate median values for each continent for the initial year (1960)
median_values = df_rural.groupby("Continent")[str(1960)].median().sort_values(ascending=False).index

# Add "world" to the list of continents
median_values = ["World"] + list(median_values)

# Create a new DataFrame that includes data for all countries (world)
df_world = df_rural.copy()
df_world["Continent"] = "World"

# Define the layout of the dashboard for init
app.layout = html.Div(
    id='app-container',
    style={
        "display": "flex",
        "flex-direction": "column",
        "align-items": "center",
        "justify-content": "space-between",
        "backgroundColor": "white",
        "padding": "5px",
        "margin": "0",
        "width": "100%",
        "height": "99vh",
        "overflow-y": "hidden",
        "box-sizing": "border-box",
        "font-family": "Arial",
    },
    children=[
        ## Hidden Divs
        html.Div(id='selected-year', style={'display': 'none'}),
        html.Div(id='hovered-country', style={'display': 'none'}),
        ######### App content #########
        # Contains the title and some infoSS
        html.Div(
            id='header-container',
            style={
                "backgroundColor": "black",
                "padding": "10px",
                "width": "100%",
                "height": "8%",
                "box-sizing": "border-box",
                "display": "flex",
                "flex-direction": "row",
                "align-items": "center",
                "justify-content": "center",
            },
            children=[
                html.Div(
                    id='title-container',
                    style={
                        "backgroundColor": "black",
                        "flex": "2",
                        "margin": "2px 0",
                    },
                    children=[
                        html.H1("Interactive Geo Data Visualization", style={'text-align': 'center','color':'white'}),
                        html.H4("WorldBank Dataset: Agriculture & Rural Development ", style={'text-align': 'center','color':'white'}),
                    ]
                ),
                html.Div(
                    id='info-container',
                    style={
                        "backgroundColor": "black",
                        "flex": "1",
                        "flex-direction":"column",
                        "justify-content":"center",
                        "align-items":"center",
                        "font-weight": "bold",
                        "font-size":"12px",
                    },
                    children=[
                        html.P("Animation playback speed can be adjusted, slow connection or pc it is advised to use above 500ms refresh rate.", style={'text-align': 'right','color':'white'}),
                        html.P("Hovering over the boxplot will show the countries of a continent. Hover over World to see all countries again.", style={'text-align': 'right','color':'white'}),
                        html.P("Hovering over the choropleth will show the country in the barplot and highlight it's position in the histogram.", style={'text-align': 'right','color':'white'}),
                    ]
                ),
            ]
        ),
        # End of title and info container

        # Contains the graphs
        html.Div(
            id='content-container',
            style={
                "backgroundColor": "white",
                "padding": "5px",
                "width": "100%",
                "height": "100%",
                "box-sizing": "border-box",
                "display": "flex",
                "flex-direction": "column",
                "justify-content": "space-between",
            },
            children=[
                html.Div(
                    id='top-container',
                    style={
                        "backgroundColor": "white",
                        "flex": "6",
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "space-between",
                    },
                    children=[
                        html.Div(
                            id='left-graph',
                            style={
                                "backgroundColor": "white",
                                "flex": "3",
                                "padding": "5px",
                                "border": "1px solid black",
                            },
                            children=[
                                dcc.Graph(
                                id='chloropleth-map',
                                figure=px.choropleth(
                                    df_rural,
                                    locations='Country Code',
                                    color=str(min_year),  # Initialize with the minimum year
                                    hover_name='Country Name',
                                    hover_data=['Continent','Country Code'],
                                    title=f'Choropleth Map of Rural Percentage ({str(min_year)})',
                                    projection='equirectangular',
                                    color_continuous_scale='Viridis',
                                    color_continuous_midpoint=50,
                                    range_color=[0, 100],
                                ).update_layout(
                                    title={'x': 0.5},
                                    margin={'l': 0, 'r': 0, 't': 50, 'b': 0},
                                    hovermode='closest',
                                    dragmode=False
                                ),
                                    # Chloropleth Style
                                    style={'flex': '1', 'width': '100%','height':'100%'}
                                )
                            ]
                        ),
                        html.Div(
                            id='right-graph',
                            style={
                                "backgroundColor": "white",
                                "flex": "1",
                                "padding": "5px",
                                "border": "1px solid black",
                            },
                            children=[
                                dcc.Graph(
                                    id='barplot',
                                    figure=px.bar(
                                        df_rural[df_rural['Country Name'] == 'China'] \
                                            .melt(
                                                id_vars=['Country Name', 'Country Code', 'Continent', \
                                                         'Indicator Name', 'Indicator Code'], 
                                                var_name='Year', 
                                                value_name='Rural population Pcnt.'),
                                        x='Rural population Pcnt.', 
                                        y='Year', 
                                        hover_data=['Country Code', 'Country Name','Rural population Pcnt.','Year'],
                                        orientation='h',
                                        color='Rural population Pcnt.',
                                        color_continuous_scale='Bluered_r',
                                        color_continuous_midpoint=50,
                                        range_color=[0, 100],
                                        title=f'Hover-over: Rural Population (% of Total Population) in {"CHN"}',
                                    ).update_layout(
                                        dragmode=False,
                                        title={'x': 0.5},
                                        showlegend=False,
                                    ),
                                    # Bar Style
                                    style={'flex': '1', 'width': '100%','height':'100%'}
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    id='bottom-container',
                    style={
                        "backgroundColor": "white",
                        "flex": "4",
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "space-between",
                    },
                    children=[
                        html.Div(
                            id='bottom-left-graph',
                            style={
                                "backgroundColor": "white",
                                "flex": "1",
                                "padding": "5px",
                                "border": "1px solid black",
                            },
                            children=[
                                dcc.Graph(
                                    id='box',
                                    figure=px.box(
                                        pd.concat([df_world,df_rural]),
                                        x="Continent", 
                                        y=str(1960),
                                        color='Continent',
                                        category_orders={"Continent": median_values},  # Order by descending medians
                                        title=f'Box Plot of Rural Percentage (1960)',
                                    ).update_layout(
                                        title={'x': 0.5},
                                        dragmode=False,
                                    ).update_traces(
                                        showlegend=False
                                    ).update_yaxes(
                                        title="Population % Living in Rural Areas",
                                        range=[0, 100],
                                    ).update_xaxes(
                                        title="Grouped by Continent",
                                    ),
                                    # Box style
                                    style={'flex': '1', 'width': '100%','height':'100%'}
                                )
                            ]
                        ),
                        html.Div(
                            id='bottom-right-graph',
                            style={
                                "backgroundColor": "white",
                                "flex": "1",
                                "padding": "5px",
                                "border": "1px solid black",
                            },
                            children=[
                                dcc.Graph(
                                    id='histogram',
                                    figure=go.Figure(
                                        data=[
                                            go.Histogram(
                                                x=df_rural[str(1960)],
                                                nbinsx=20,
                                                xbins=dict(
                                                    start=0,
                                                    end=100,
                                                    size=5  # Set the bin size to 5 to create 5% intervals
                                                ),
                                                name='All Countries',
                                                marker=dict(color='lightgray', line=dict(width=2)),  # Change the color and line properties as needed
                                                opacity=0.7,
                                            ),
                                        ],
                                        layout=dict(
                                            title=f'Histogram Bins of Rural Percentage Globally (1960)',
                                            xaxis_title="Percentage Range Bins",
                                            yaxis_title="Number of Countries",
                                            yaxis=dict(range=[0, 30]),  # Set the y-axis range to [0, 30]
                                            xaxis=dict(range=[0, 100],dtick=5),  # Set the x-axis range to [0, 100] and dtick to 5
                                            dragmode=False,
                                        )
                                    ),
                                    style={'flex': '1', 'width': '100%', 'height': '100%'}
                                )
                            ]
                        ),

                    ]
                ),
            ]
        ),
        # End of graph container

        # Contains the play controlls
        html.Div(
            id='footer-container',
            style={
                "backgroundColor": "black",
                "padding": "10px 0",
                "width": "100%",
                "display": "flex",
                "flex-direction": "row",
                "justify-content": "space-between",
                "borderRadius": "5px",
            },
            children=[
                html.Div(
                    id='footer-child-1',
                    style={
                        "backgroundColor": "black",
                        "flex": "none", 
                        "width": "100px",
                        "margin": "0 10px",
                    },
                    children=[
                        # Play/pause button
                        html.Button(
                            "Play",
                            id="play-button",
                            n_clicks=0,
                            style={'flex': '1', 'width': '100%','height':'100%'}
                        ),
                    ]
                ),
                html.Div(
                    id='footer-child-2',
                    style={
                        "backgroundColor": "black",
                        "flex": "8",
                        "margin": "0 10px", 
                        "fontWeight": "bold",
                        "borderRadius": "5px",
                    },
                    children=[
                        # Slider for selecting years
                        dcc.Slider(
                            id='year-slider',
                            min=min_year,
                            max=max_year,
                            step=1,
                            value=min_year,
                            marks={str(year): str(year) for year in years},
                            included=False,
                        ),
                        # Interval component for animation
                        dcc.Interval(
                            id='animation-interval',
                            interval=300,  # Interval in milliseconds
                            disabled=True,  # Initially disabled
                        ),
                        html.Div(
                            style={
                                'color':'white',
                                'textAlign':'center',
                            },
                            children=[
                            html.Label('Animation Speed (Milliseconds) - Lower is Faster'),
                            dcc.Slider(
                                id='speed-slider',
                                min=50,  # Minimum interval value in milliseconds
                                max=800,  # Maximum interval value in milliseconds
                                step=50,  # Step size for the slider
                                value=500,  # Initial value of the interval in milliseconds
                            ),
                        ]),
                    ]
                ),
            ]
        )
        # End of play controls container
    ##### End of App content ######
])

# In your callback function
@app.callback(
    Output('year-slider', 'value'),  # Output: Update the value of the year slider
    Input('animation-interval', 'n_intervals'),  # Input: n_intervals from animation interval
    Input('year-slider', 'value'),  # Input: Current value of the year slider
    Input('play-button', 'n_clicks'),  # Input: Clicks on the play button
    Input('year-slider', 'max'),  # Input: Maximum value of the year slider
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_selected_year(n_intervals, slider_value, play_button_clicks, max_year):
    if play_button_clicks % 2 == 1:
        # Play button is clicked, start animation
        if slider_value < max_year:
            slider_value += 1
        else:
            slider_value = min_year
    return slider_value

# Callback to update the hovered country in the hidden div
@app.callback(
    Output('hovered-country', 'children'),  # Output: Update the content of the hidden div for hovered country
    Input('chloropleth-map', 'hoverData'),  # Input: Hover data from the choropleth map
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_hovered_country(hover_data):
    if hover_data is not None and 'points' in hover_data:
        country_name = hover_data['points'][0]['hovertext']
        return country_name
    else:
        return ''

# Callback to update the box hover data based on the selected continent
@app.callback(
    Output('box', 'hoverData'),  # Output: Update hover data for the box plot
    Input('box', 'selectedData'),  # Input: Selected data in the box plot
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_box_hover(selected_data):
    if selected_data is not None and 'points' in selected_data:
        selected_continent = selected_data['points'][0]['x']
        if selected_continent == "World":
            # Return hover data for "World" (all countries)
            selected_continent = None

        return {'points': [{'x': selected_continent}]}
    else:
        return {'points': []}

# Define callback functions
@app.callback(
    Output('chloropleth-map', 'figure'),  # Output: Update the choropleth map figure
    Input('year-slider', 'value'),  # Input: Selected year from the year slider
    Input('box', 'hoverData'),  # Input: Hover data from the box plot
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_choropleth_animation(selected_year, box_hover_data):
    # Initialize the filtered DataFrame with all countries
    filtered_df = df_rural.copy()

    if box_hover_data is not None and 'points' in box_hover_data:
        selected_continent = box_hover_data['points'][0]['x']
        if selected_continent != "World":
            # Filter the DataFrame to get data for the selected continent and year
            filtered_df = df_rural[df_rural['Continent'] == selected_continent]

    fig = px.choropleth(
        filtered_df,
        locations='Country Code',
        color=str(selected_year),
        hover_name='Country Name',
        hover_data=['Continent', 'Country Code'],
        title=f'Choropleth Map of Rural Percentage ({str(selected_year)})',
        projection='equirectangular',
        color_continuous_scale='Viridis',
        color_continuous_midpoint=50,
        range_color=[0, 100],
    ).update_layout(
        title={'x': 0.5},
        margin={'l': 0, 'r': 0, 't': 50, 'b': 0},
        hovermode='closest',
        dragmode=False,
    )

    return fig

@app.callback(
    Output('barplot', 'figure'),  # Output: Update the bar plot figure
    Input('hovered-country', 'children'),  # Input: Hovered country name
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_barplot_animation(hover_country):
    # Update the figure with the new year
    bar_fig = px.bar(
        df_rural[df_rural['Country Name'] == hover_country].melt(
            id_vars=['Country Name', 'Country Code', 'Continent', 'Indicator Name', 'Indicator Code'],
            var_name='Year',
            value_name='Rural population Pcnt.'
        ),
        x='Rural population Pcnt.',
        y='Year',
        hover_data=['Country Code', 'Country Name', 'Rural population Pcnt.', 'Year'],
        orientation='h',
        color='Rural population Pcnt.',
        color_continuous_scale='Bluered_r',
        color_continuous_midpoint=50,
        range_color=[0, 100],
        title=f'Hover-over: Rural Population (% of Total Population) in {hover_country}',
    ).update_layout(
        title={'x': 0.5},
        dragmode=False,
        showlegend=False,
    ).update_yaxes(autorange="reversed")

    return bar_fig

@app.callback(
    Output('box', 'figure'),  # Output: Update the box plot figure
    Input('year-slider', 'value'),  # Input: Selected year from the year slider
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_box_animation(selected_year):
    # Calculate median values for each continent
    median_values = df_rural.groupby("Continent")[str(selected_year)].median().sort_values(ascending=False).index

    # Add "world" to the list of continents
    median_values = ["World"] + list(median_values)

    # Create a DataFrame that includes data for all countries (world)
    df_world = df_rural.copy()
    df_world["Continent"] = "World"

    # Update the figure with the new year
    box_fig = px.box(
        pd.concat([df_rural, df_world]),  # Concatenate the data for all continents including "world"
        x="Continent",
        y=str(selected_year),
        color='Continent',
        category_orders={"Continent": median_values},  # Order by descending medians
        title=f'Scatter Plot of Rural Percentage ({str(selected_year)})',
    ).update_layout(
        title={'x': 0.5},
        hovermode='closest',
        dragmode=False
    ).update_traces(
        showlegend=False
    ).update_yaxes(
        range=[0, 100],
        title="Population % Living in Rural Areas",
    ).update_xaxes(
        title="Grouped by Continent",
    )

    return box_fig

# Callback to update the histogram based on the selected country and year slider
@app.callback(
    Output('histogram', 'figure'),  # Output: Update the histogram figure
    Input('hovered-country', 'children'),  # Input: Hovered country name
    Input('year-slider', 'value'),  # Input: Selected year from the year slider
    Input('year-slider', 'drag_value'),  # Input: Drag value of the year slider
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def update_histogram_on_hover(selected_country, selected_year, drag_value):
    # Determine whether the callback was triggered by a hover event or slider value change
    ctx = dash.callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_component == 'year-slider':
        # Slider value changed, update the histogram based on the selected year
        if selected_country is None:
            selected_country = "All Countries"  # Default country when slider changes

    elif triggered_component == 'chloropleth-map':
        # Hover event, use the stored selected country
        if selected_country is None:
            selected_country = "All Countries"  # Default country when no hover event

    # Create a histogram trace for all countries with consistent bin settings
    histogram_trace = go.Histogram(
        x=df_rural[str(selected_year)],  # Update this to the appropriate year or column
        nbinsx=20,
        xbins=dict(
            start=0,
            end=100,
            size=5  # Set the bin size to 5 to create 5% intervals
        ),
        name='All Countries',
        marker=dict(line=dict(width=2)),  # Change the color and line properties as needed
        opacity=0.7,
    )

    # Create a list of shapes to highlight the selected country's bin
    shapes = []

    if selected_country != "All Countries":
        # Filter the DataFrame to get data for the selected country
        filtered_df = df_rural[df_rural['Country Name'] == selected_country]

        # Calculate the bin edges and highlight the selected country's bin
        hist, edges = np.histogram(filtered_df[str(selected_year)], bins=20, range=[0, 100])
        selected_bin_index = np.digitize(filtered_df.iloc[0][str(selected_year)], edges) - 1

        # Calculate the center of the selected bin
        center = (edges[selected_bin_index] + edges[selected_bin_index + 1]) / 2

        # Highlight the selected bin with a different color
        shapes.append(
            dict(
                type='rect',
                x0=center - 2.5,  # Adjust the x0 and x1 to center the shape
                x1=center + 2.5,
                y0=0,
                y1=max(hist),
                xref='x',
                yref='y',
                fillcolor='red',  # Change the color as needed
                opacity=1,
                layer='above',  # Place the shape above the bars
                line=dict(width=2, color='black'),  # Add a black outline
            )
        )

    # Create the histogram figure with the histogram trace and shapes
    histogram_fig = go.Figure(data=[histogram_trace], layout=dict(shapes=shapes))
    histogram_fig.update_layout(
        title=f'Histogram Bins of Rural Percentage Globally ({selected_year}) Hover Country: {selected_country}',
        xaxis_title="Percentage Range Bins",
        yaxis_title="Number of Countries",
        dragmode=False,
        yaxis=dict(range=[0, 30]),  # Set the y-axis range to [0, 30]
        xaxis=dict(range=[0, 100],dtick=5),  # Set the x-axis range to [0, 100]
    ).update_layout(
        title={'x': 0.5},
    )

    return histogram_fig

@app.callback(
    Output('play-button', 'children'),  # Output: Update the play button label
    Output('animation-interval', 'disabled'),  # Output: Enable/disable the animation interval
    Input('play-button', 'n_clicks'),  # Input: Clicks on the play button
    Input('animation-interval', 'disabled'),  # Input: Animation interval status (disabled/enabled)
    prevent_initial_call=True  # Prevent initial call when the app starts
)
def toggle_play_pause(n_clicks, interval_disabled):
    if n_clicks % 2 == 1:
        return "Pause", False
    else:
        return "Play", True

# Set animation speed
@app.callback(
    Output('animation-interval', 'interval'),  # Output: Update the animation interval
    Input('speed-slider', 'value')  # Input: Speed value from the speed slider
)
def update_animation_interval(speed_value):
    interval_value = speed_value
    return interval_value

# Display the app within the Jupyter Notebook cell
# display(HTML("<style>.container { width:100% !important; font-family: Ubuntu;}</style>"))
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='80')