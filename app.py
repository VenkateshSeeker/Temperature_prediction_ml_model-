import joblib
import pandas as pd
import streamlit as st

model = joblib.load("feelslike_model.pkl")
st.title("Feels Like Temperature Predictor")
st.write("Enter weather conditions to predict feels-like temperature:")

temp = st.number_input("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 3)
clouds = st.slider("Cloudiness (%)", 0, 100, 40)
input_data = pd.DataFrame([[temp, humidity, wind_speed, clouds]],columns=["temp", "humidity", "wind_speed", "clouds"])
feels_like_pred = model.predict(input_data)[0]
st.success(f"Predicted Feels Like Temperature: {feels_like_pred:.2f} °C")
st.subheader(" What is 'Feels Like' Temperature?")
st.markdown("""
The **'feels like' temperature** is not always the same as the actual air temperature.  
It is how hot or cold the weather **actually feels to humans**, and depends on factors like:

-  **Humidity**: High humidity makes it feel **hotter** (because sweat doesn't evaporate well).  
-  **Wind Speed**: Strong winds make it feel **colder** (this is called *wind chill*).  
-  **Cloudiness**: Clouds can block sunlight, making it feel cooler, or trap heat at night, making it feel warmer.  

### The Science Behind It
- In **hot weather**, meteorologists use the **Heat Index** formula, which combines **temperature and humidity**.  
- In **cold weather**, they use the **Wind Chill Index**, which combines **temperature and wind speed**.  

Your weather app (and this predictor) uses these factors to estimate the **'feels like' temperature**.
""")
