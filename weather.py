import requests 
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

api_key = '0b33789d15677151202a984a2f5b80f1' 

lat = 18.184135
lon = 74.610764

api_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(api_url)
weather_data = response.json()

# Extract relevant weather attributes using list comprehension
timestamps = [pd.to_datetime(item['dt'], unit='s') for item in weather_data['list']]
temperatures = [item['main']['temp'] for item in weather_data['list']]
humidity = [item['main']['humidity'] for item in weather_data['list']]
wind_speed = [item['wind']['speed'] for item in weather_data['list']]
weather_description = [item['weather'][0]['description'] for item in weather_data['list']]

# Create a pandas DataFrame with the extracted weather data
weather_df = pd.DataFrame({
    'Timestamp': timestamps,                           
    'Temperature': temperatures,                           
    'humidity': humidity,                           
    'wind_speed': wind_speed,
    'weather_description': weather_description
})
                         
# Set the Timestamp column as the DataFrame's index
weather_df.set_index('Timestamp', inplace=True)

weather_df.fillna(0, inplace=True) # Replace missing values with 0 or appropriate value

# Convert temperature from Kelvin to Celsius
weather_df['Temperature'] = weather_df['Temperature'].apply(lambda x: x - 273.15 if isinstance(x, float) else x)

max_temp = weather_df['Temperature'].max()
print(f"Maximum Temperature - {max_temp}")
min_temp = weather_df['Temperature'].min()
print(f"Minimum Temperature - {min_temp}")

print(weather_df)

daily_mean_temp = weather_df['Temperature'].resample('D').mean()
daily_mean_humidity = weather_df['humidity'].resample('D').mean()
daily_mean_wind_speed = weather_df['wind_speed'].resample('D').mean()

# Plot the mean daily temperature over time (Line plot)
plt.figure(figsize=(10, 6))
daily_mean_temp.plot(color='red', linestyle='-', marker='o')
plt.title('Mean Daily Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Plot the mean daily humidity over time (Bar plot)
plt.figure(figsize=(10, 6))
daily_mean_humidity.plot(kind='bar', color='blue')
plt.title('Mean Daily Humidity')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.show()    

# Plot the relationship between temperature and wind speed (Scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(weather_df['Temperature'], weather_df['wind_speed'], color='green')
plt.title('Temperature vs. Wind Speed')
plt.xlabel('Temperature (°C)')
plt.ylabel('Wind Speed (m/s)')
plt.grid(True)
plt.show()

# Heatmap (fixed to plot correlation matrix instead of raw 40-row values)
plt.figure(figsize=(8, 6))
sns.heatmap(weather_df[['Temperature', 'humidity', 'wind_speed']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Create a scatter plot to visualize the relationship between temperature and humidity
plt.figure(figsize=(10, 6))
plt.scatter(weather_df['Temperature'], weather_df['humidity'], color='purple')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Temperature vs Humidity Scatter Plot')
plt.grid(True)
plt.show()
