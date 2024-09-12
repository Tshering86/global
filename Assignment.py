#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random
import altair as alt
alt.data_transformers.enable('json')
from matplotlib import pyplot as plt                  
import seaborn as sns                                 
sns.set_theme(style="white", palette="muted") 
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

import warnings                                       # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")


# In[93]:


get_ipython().system('pip install wordcloud')


# In[12]:


pip install kaggle


# In[2]:


get_ipython().system('kaggle datasets download -d nelgiriyewithana/global-weather-repository')


# In[2]:


import zipfile

# Unzip the dataset
with zipfile.ZipFile('global-weather-repository.zip', 'r') as zip_ref:
    zip_ref.extractall('weather_data')
    


# In[3]:


# Read the CSV
df = pd.read_csv('weather_data/GlobalWeatherRepository.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


# Counties Weather Condition
barchart = px.histogram(df, x="condition_text", title="Weather Condition", color="condition_text", height=600, width=1200)
barchart.update_xaxes(title="Weather Condition")
barchart.update_yaxes(title="Count")

# Add data labels on top of bars
barchart.update_traces(texttemplate="Count %{y:2f}", textposition="outside")
# Add interactive legends
barchart.update_layout(showlegend=True)
# Customize color palette
barchart.update_traces(marker=dict(line=dict(color='rgb(0,0,0)', width=0.3)))
# Customize hover information
barchart.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y:2f}")
# Customize the legend title
barchart.update_layout(legend_title_text="Weather Condition")

barchart.show()


# In[10]:


zone = ['Asia/Kabul', 'Europe/Tirane', 'Africa/Algiers', 'Europe/Andorra','Africa/Luanda', 'America/Antigua','America/Argentina/Buenos_Aires','Australia/Sydney', 
        'Europe/Vienna','Asia/Bahrain', 'Asia/Dhaka','Europe/Brussels', 'America/Belize','America/Manaus', 'Asia/Brunei','Africa/Ouagadougou','Atlantic/Cape_Verde',
        'Asia/Phnom_Penh', 'Africa/Douala', "Asia/Bangkok"]


# In[11]:


weather_zones = df.query("timezone in @zone")
# Create the Sunburst chart
fig = px.sunburst(
    weather_zones, 
    path=['timezone', 'country'], # Define the hierarchical path
    values='temperature_celsius', # Add values
    height=600,
    color='temperature_celsius',  # Color based on temperature
    color_continuous_scale='thermal', # Use the Temps color scale
    color_continuous_midpoint=np.mean(weather_zones['temperature_celsius']), # Set color midpoint to the median temperature
    labels={'temperature_celsius': 'Temperature (°C)'},  # Customize the colorbar label
)

# Set title
fig.update_layout(
    title="Average Temperature by Time Zone and Country",
)

fig.show()


# In[12]:


fig = px.scatter_polar(
    weather_zones,
    r="wind_kph", # Radial distance represents wind speed in kilometers per hour
    theta="wind_direction", # Angular position represents wind direction
    color="temperature_celsius", # Color represents temperature in degrees Celsius
    color_continuous_scale = "viridis", # Use viridis color scale
    title="Wind Speed and Wind Direction",
    labels={'temperature_celsius': 'Temperature (°C)'}, # Customize the colorbar label
    height=600, # Add height to the polar chart
    hover_name="location_name", # Add location names as hover text
)

fig.show()


# In[13]:


fig = px.scatter(
    weather_zones,
    x="temperature_celsius",
    y="humidity",
    color="temperature_celsius", 
    color_continuous_scale = "tempo",
    size='wind_kph', # Size of data points based on wind speed in kph
    hover_name='country', # Add hover value
    hover_data=['location_name'],  # Additional data to show on hover (location_name)
    labels=
        {
            'temperature_celsius': 'Temperature (°C)',
            'humidity': 'Humidity',
            'wind_kph': 'Wind speed (kph)',
            'country': 'Country',
            'condition_text': 'Weather Condition',
            'location_name': 'City'
        },
    title="Temperature and Humidity with Wind Speed",
    height=600,
    )

fig.update_xaxes(title="Temperature (°C)")
fig.update_yaxes(title="Humidity")


fig.show()


# In[14]:


# Group the weather condition and calculate the mean of temperature
weather_status = weather_zones.groupby('condition_text')['temperature_celsius'].agg(['mean']).reset_index()
# Rename the columns
weather_status = weather_status.rename(columns={'mean': 'temperature'})
weather_status


# In[15]:


fig = px.pie(
    weather_status,
    names='condition_text',
    values='temperature',
    title='Weather Condition And Temperature',
    height=600,
    labels=(
        {
            'condition_text': 'Weather Condition',
            'temperature': 'Temperature (°C)'
        }
    )
)

# Add labels inside the pie chart section
fig.update_traces(textposition='inside', textinfo='label+percent')

fig.show()


# In[16]:


#Creating and Analysing a new attribute - Continent/Region

df[['Continent/Region','City','Extra']] = df['timezone'].str.split('/',expand=True)
df.drop(['City','Extra'],axis=1,inplace=True) 

df_Continent = df.groupby('Continent/Region').agg({'last_updated':'count'}).sort_values(by='last_updated',ascending=False).reset_index(drop=False)

#Canvas
fig, ax = plt.subplots(figsize=(9, 9))

#Plotting
plt.pie(x=df_Continent['last_updated'],
        labels = df_Continent['Continent/Region'],
        wedgeprops=dict({'width': 0.2}),
        autopct='%.1f%%')
plt.title('Continent/Region Split - Weather Recording Frequency', fontsize=16)
plt.tight_layout()
plt.show()


# In[17]:


# Group the moon phases and calculate the mean of moon illumination
weather_zones_moon = weather_zones.groupby('moon_phase')['moon_illumination'].agg(['mean']).reset_index()
# Rename the columns
weather_zones_moon = weather_zones_moon.rename(columns={'mean': 'moon_illumination'})
weather_zones_moon


# In[18]:


fig = px.pie(
    weather_zones_moon,
    names='moon_phase',
    values='moon_illumination',
    title='Moon Phase and Moon Illumination',
    height=600,
    hole=0.4,
    labels=(
        {
            'moon_phase': 'Moon Phase',
            'moon_illumination': 'Moon Illumination (%)'
        }
    ),
)

#Add labels inside the pie chart section
fig.update_traces(textposition='inside', textinfo='label+percent')

fig.show()


# In[19]:


temperature_max = df['temperature_celsius'].max()
temperature_min = df['temperature_celsius'].min()
temperature_mean = df['temperature_celsius'].mean()


# In[20]:


df['last_updated'] = pd.to_datetime(df['last_updated'])

last_updated = df['last_updated'].max().strftime('%Y-%m-%d %H:%M:%S')

average_temp_by_country_year = df.groupby([df['last_updated'].dt.year, 'country'])['temperature_celsius'].mean().reset_index()
average_temp_by_country_year.columns = ['year', 'country', 'average_temperature']

print(average_temp_by_country_year)


# In[21]:


fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value= temperature_mean,
    title={'text' : 'Temperature (°C)'},
    gauge= { 'axis' : {'range' : [None, 50]},
            'bar': {'color': "#008000"},
            'steps' : [
                {'range' : [0, 10], 'color' : "seashell"}, 
                {'range' : [10, 20], 'color' : "lightblue"},
                {'range' : [20, 30], 'color' : "lightgreen"},
                {'range' : [30, 40], 'color' : "sandybrown"},
                {'range' : [40, 50], 'color' : "tomato"}
            ],
            'threshold' : {
                'line' : {'color' : 'red', 'width' : 5},
                'thickness' : 0.75,
                'value' : temperature_mean,
            }
    }
))

fig.update_layout(
    title = "Average global temperature <br>(updated " + last_updated + ")"
)

fig.show()


# In[22]:


fig = px.scatter_mapbox(
    df,
    lat="latitude",  # Latitude data column
    lon="longitude", # Longitude data column 
    color="temperature_celsius", # Color data for points
    color_continuous_scale=px.colors.cyclical.IceFire, # Color scale
    hover_name= 'location_name', # Add hover value
    size="humidity", # Based on size data will change
    size_max=7,  # Maximum size for points
    labels=( 
        {
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'temperature_celsius' : 'Temperature (°C)',
            'humidity': 'Humidity'
        }),
    height=600,
    width=1000
    )

fig.update_layout(
    mapbox_style='open-street-map', # Map style
    title="Temperature and Humidity ",  # Title of the map
    hovermode='closest',  # Hover mode for interactivity
    mapbox=dict(
        bearing=0, # Bearing of the map
        center=go.layout.mapbox.Center(
            lat=47, # Center latitude
            lon=12 # Center longitude
        ),
        pitch=0, # Map pitch
        zoom=4 # Initial map zoom level
    )
)

fig.show()


# In[23]:


fig = px.choropleth(
    average_temp_by_country_year,
    locations="country",
    locationmode="country names",
    color="average_temperature",
    hover_name="country",
    color_continuous_scale=px.colors.sequential.Viridis,
    projection="natural earth",  # Use a valid projection type
    title="Global Average Temperaturas" )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[24]:


top_countries1 = average_temp_by_country_year.groupby('country')['average_temperature'].mean().nlargest(5).reset_index()
top_countries1 = top_countries1.sort_values(by='average_temperature')

top_countries2 = average_temp_by_country_year.groupby('country')['average_temperature'].mean().nsmallest(5).reset_index()
top_countries2 = top_countries2.sort_values(by='average_temperature')

# Combine the hottest and coldest countries
top_countries = pd.concat([top_countries1, top_countries2])

# Create the bar chart
fig = px.bar(
    top_countries, 
    x='average_temperature', 
    y='country', 
    orientation='h', 
    title='Top 5 Hottest and Coldest Countries', 
    labels={'average_temperature': 'Average Temperature (°C)', 'country': 'Countries'},
    text=top_countries['average_temperature'].round(2).astype(str) + ' °C',
    color ='average_temperature' 
)


# Show the plot
fig.show()


# In[27]:


# Sample the dataset
random.seed(10)
df1 = df.sample(3500)

# Create a dropdown for country selection
input_dropdown = alt.binding_select(options=list(df1['country'].unique()), name='Select Country')
selected_points = alt.selection_single(fields=['country'], bind=input_dropdown)

# Chart 1: Scatter plot for temperature (filtered by country selection)
chart1 = alt.Chart(df1).mark_point(color='green').encode(
    x='temperature_fahrenheit',
    tooltip=["country", "temperature_fahrenheit"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title="Temperature (Fahrenheit) by Country"
)
chart1

# Chart 2: Trail plot for Carbon Monoxide (filtered by country selection)
chart2 = alt.Chart(df1).mark_trail(color='green').encode(
    x='air_quality_Carbon_Monoxide',
    tooltip=["country", "air_quality_Carbon_Monoxide"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title="Carbon Monoxide Levels by Country"
)
chart2

# Chart 3: Tick plot for PM10 (filtered by country selection)
chart3 = alt.Chart(df1).mark_tick(color='green').encode(
    x='air_quality_PM10',
    tooltip=["country", "air_quality_PM10"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title="PM10 Levels by Country"
)
chart3

# Chart 4: Bar plot for weather condition vs wind speed (filtered by country selection)
c2 = alt.Chart(df1).mark_bar(color="green").encode(
    y="mean(wind_mph)",
    x=alt.X("condition_text", sort=alt.EncodingSortField(field='wind_mph', op='mean')),
    tooltip=["country", "wind_mph"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title="Weather Condition Vs Average Wind (mph)",
    width=700,
    height=500
)
c2

# Chart 5: Bar plot for humidity (filtered by country selection)
chart5 = alt.Chart(df1).mark_bar(color="green").encode(
    x=alt.X('humidity', bin=True),
    y='count()',
    tooltip=["country", "humidity"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title="Humidity Bar Plot by Country",
    width=700,
    height=500
)
chart5

# Chart 6: Scatter plot for visibility vs gust (filtered by country selection)
interval = alt.selection_interval()
chart6 = alt.Chart(df1).mark_point().encode(
    x="visibility_miles",
    y="gust_mph",
    color=alt.condition(interval, 'gust_mph', alt.value('lightgray')),
    tooltip=["visibility_miles", "gust_mph", "country"]
).add_selection(
    interval
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title="Visibility (miles) Vs Gust (mph) by Country",
    width=700,
    height=500
)
chart6

# Filter dataset based on last_updated date
df1["last_updated"] = df1["last_updated"].astype(str)
df2 = df1[df1["last_updated"].str.startswith("2024-09-09")]

# Chart 7: Bar plot for temperature across countries on 2024-09-09 (filtered by country selection)
chart7 = alt.Chart(df2).mark_bar(color='green').encode(
    x='country',
    y='temperature_celsius',
    tooltip=["country", "temperature_celsius"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title='Temperature in Celsius across countries (2024-09-09)',
    width=900,
    height=300
)
chart7

# Chart 8: Bar plot for AQI across countries on 2024-09-09 (filtered by country selection)
chart8 = alt.Chart(df2).mark_bar(color='green').encode(
    x='country',
    y='air_quality_us-epa-index',
    tooltip=["country", "air_quality_us-epa-index"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title='AQI across countries (2024-09-09)',
    width=900,
    height=300
)
chart8

# Chart 9: Scatter plots for AQI vs Carbon Monoxide and PM10 (filtered by country selection)
interval = alt.selection_interval()
base = alt.Chart(df1).mark_circle().encode(
    y="air_quality_us-epa-index", 
    color=alt.condition(interval, 'air_quality_PM10', alt.value('lightgray')),
    tooltip=["air_quality_Carbon_Monoxide", "air_quality_PM10", "air_quality_us-epa-index", "country"]
).add_selection(
    interval
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    title='AQI Vs Carbon Monoxide and PM10 by Country',
    width=400,
    height=300
)

scatter = base.encode(x='air_quality_Carbon_Monoxide') | base.encode(x='air_quality_PM10')
scatter

# Chart 10: Scatterplot matrix (SPLOM) (filtered by country selection)
chart = alt.Chart(df1).mark_circle(color="green", opacity=0.3).encode(
    alt.X(alt.repeat("column"), type="quantitative"),
    alt.Y(alt.repeat("row"), type="quantitative"),
    tooltip=["precip_in", "wind_mph", "humidity", "pressure_mb", "gust_mph", "country"]
).add_selection(
    selected_points
).transform_filter(
    selected_points
).properties(
    width=125,
    height=125
).repeat(
    row=["precip_in", "wind_mph", "humidity", "pressure_mb", "gust_mph"],
    column=["precip_in", "wind_mph", "humidity", "pressure_mb", "gust_mph"]
).interactive()

chart


# In[26]:


# Example stopwords list, modify as needed
stop_words_list = set(STOPWORDS)

# Generate frequency counts from the 'country' column
counts = Counter(df["country"].dropna().apply(lambda x: str(x)))

# Generate the word cloud
wcc = WordCloud(
    background_color="black",
    width=1600, height=800,
    max_words=2000,
    stopwords=stop_words_list
)
wcc.generate_from_frequencies(counts)

# Display the word cloud
plt.figure(figsize=(10, 5), facecolor='k')
plt.imshow(wcc, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




