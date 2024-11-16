import streamlit as st
import pandas as pd
import numpy as np
#import sys
#import io

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("project/pages/hour.csv")

hour_df = load_data()
# Owner's note: While it is true that the transformations could be skipped by applying the transformations in a notebook and exporting the cleaned dataset
# we opted in putting the transformations inside the script to ensure that with a new "raw" dataset, considering same columns and attributes, the project is scalable.

# Dictionary of holidays in 2024 with holiday names as keys and dates as values
holidays_2024 = {
    "New Year's Day": '2024-01-01',
    "Martin Luther King Jr. Day": '2024-01-15',
    "Presidents' Day": '2024-02-19',
    "Memorial Day": '2024-05-27',
    "Independence Day": '2024-07-04',
    "Labor Day": '2024-09-02',
    "Columbus Day": '2024-10-14',
    "Veterans Day": '2024-11-11',
    "Thanksgiving Day": '2024-11-28',
    "Christmas Day": '2024-12-25'
}

# Convert holiday dates to datetime format for easy comparison
holidays_2024 = {k: pd.to_datetime(v) for k, v in holidays_2024.items()}

# Function to check if a date is a holiday and return the holiday name if it is
def get_holiday_name(date):
    for holiday, holiday_date in holidays_2024.items():
        if date == holiday_date.date():  # Convert holiday_date to .date() for comparison
            return holiday
    return None


# Function to filter and calculate features for the selected date and hour
def calculate_previous_features(df, selected_month, selected_day, selected_hour):
    # Filter data for December 2011 and all of 2012
    df['dteday'] = pd.to_datetime(df['dteday'])
    # Get the maximum year in the 'dteday' column
    max_year = df['dteday'].dt.year.max()
    filtered_df = df[(df['dteday'].dt.year == max_year) | ((df['dteday'].dt.year == max_year-1) & (df['dteday'].dt.month == 12))]

    # Find the matching row in the dataset
    matching_row = filtered_df[
        (filtered_df['dteday'].dt.month == selected_month) &
        (filtered_df['dteday'].dt.day == selected_day) &
        (filtered_df['hr'] == selected_hour)
    ]

    if matching_row.empty:
        st.warning("No data found for the selected date and hour.")
        return None, None, None

    index = matching_row.index[0]

    # Calculate Previous Count
    previous_count = filtered_df.loc[index - 1, 'cnt'] if index - 1 in filtered_df.index else None

    # Calculate Previous Shift Mean (average of the previous 6 hours)
    previous_shift_mean = round(filtered_df['cnt'].iloc[max(0, index - 6):index].mean(),0)

    # Calculate Previous Day (count at the same hour on the previous day)
    previous_day = filtered_df.loc[index - 24, 'cnt'] if index - 24 in filtered_df.index else None

    return previous_count, previous_shift_mean, previous_day

def show_simulation():
    st.title("Simulation")
    st.markdown("""
    This page allows you to run simulations related to bike-sharing demand in Washington D.C.
    Enter the required input parameters and run the simulation to see the predicted bike demand.
    """)


    # Input fields for the simulation
    st.subheader("Simulation Inputs")
    #Check if user wants to input Historical data
    # User input to check if they know historical data
    user_knows_prev_count = st.radio("Do you know historical data?", options=["Yes", "No"],index=1)

    # Input boxes for previous hour and previous day bike count if user knows historical data
    if user_knows_prev_count == "Yes":
        user_prev_hour_count = st.number_input("Enter the number of bikes used in the previous hour:", min_value=0, max_value=1000,value=0)
        user_prev_day_count = st.number_input("Enter the number of bikes used at the same time yesterday:", min_value=0, max_value=1000,value=0)

    # IF statement to decide which data to use for the model input
    if user_knows_prev_count == "Yes":
        previous_count_user = user_prev_hour_count
        previous_day_user = user_prev_day_count

    # Slider for temperature
    temp = st.slider("Temperature (°C)", min_value=0.0, max_value=41.0, value=20.0, format="%.1f")
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, format="%.1f")
    windspeed = st.slider("Windspeed (km/h)", min_value=0.0, max_value=67.0, value=10.0, format="%.1f")



    hour = st.selectbox("Hour of the Day", options=range(0, 24))
  
    # Date input and forcing the year to be 2024
    user_date = st.date_input("Select a date (Optimized for 2024)", value=pd.Timestamp('2024-01-01'))

    # Extract month and day
    month = user_date.month
    day = user_date.day

    # Calculate previous features
    previous_count, previous_shift_mean, previous_day = calculate_previous_features(hour_df, month, day, hour)

    if previous_count is not None and user_knows_prev_count == "Yes":
        st.write(f"Bikes used one hour ago: {previous_count_user}")
        st.write(f"Bikes used one day ago (24 hours): {previous_day_user}")
    elif previous_count is not None and user_knows_prev_count == "No":
        st.write(f"Bikes used one hour ago based on historical data (2012): {previous_count}")
        st.write(f"Bikes used one day ago based on historical data (2012): {previous_day}")
    else:
        st.warning("Unable to calculate previous features due to missing data.")

    # Convert season and weather_condition to numerical codes (as done in data processing)
    season_mapping = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
    weather_mapping = {'Clear': 1, 'Misty': 2, 'Light_Rain/Snow': 3, 'Heavy_Rain/Snow': 4}

    # Determine if the selected day is a working day
    if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] and holiday == 0:
        workingday = 1  # It is a working day
    else:
        workingday = 0  # It is not a working day

    holiday_name = get_holiday_name(user_date)
    # Determine if the date is a holiday
    holiday = 1 if holiday_name else 0

    # Determine the season based on the month
    season_mapping = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
    season_code = 0 # placeholder value

    # Prepare the input DataFrame
    df = pd.DataFrame({
        'temp': [temp / 41],
        'hum': [humidity / 100],
        'windspeed': [windspeed / 67],
        'hr': [hour],
        'season': 0, # Set at 0 because it is not part of the model, if it is it would be [season_mapping[season]],
        'weathersit': 0, # Set at 0 because it is not part of the model, if it is it would be [weather_mapping[weather_condition]],
        'holiday': [holiday],
        'workingday': [workingday],
        'date' : user_date,
        'Month' : user_date.month,
        'Day' : user_date.day,
        'Previous_Count': [previous_count if previous_count is not None else 0],
        'Previous_Shift_Mean': [previous_shift_mean if previous_shift_mean is not None else 0],
        'Previous_Day': [previous_day if previous_day is not None else 0]
    })

    # Placeholding Season and Weather to 0 since they are not part of the model
    df['seasons'] = 0
    df['weather'] = 0 #weather_condition

    # Add real values for temperature, atemp, hum, and windspeed
    df['real_temp'] = df['temp'] * 41
    df['real_atemp'] = df['temp'] * 50  # Considered the same as temp
    df['real_hum'] = df['hum'] * 100
    df['real_windspeed'] = df['windspeed'] * 67

    # Placeholders for this data
    if user_knows_prev_count == "No":
        df['Previous_Count'] = previous_count  
        df['Previous_Shift_Mean'] = previous_shift_mean
        df['Previous_Day'] = previous_day  
    else:
        df['Previous_Count'] = previous_count_user
        df['Previous_Shift_Mean'] = previous_shift_mean
        df['Previous_Day'] = previous_day_user 

    # Convert 'date' column to datetime format explicitly
    df['date'] = pd.to_datetime(df['date'], errors='coerce')


    # Add Year column
    df['Year'] = df['date'].dt.year
    # Year will be forced at 2012
    df['Year'] = 2012

    # Add shift column based on hour and day
    def map_hour_to_shift(hour, day):
        if day in ['Saturday', 'Sunday']:
            if 4 <= hour < 11:
                return 'Early_Morning'
            elif 11 <= hour < 18:
                return 'Afternoon'
            else:
                return 'Late_Night'
        else:
            if 3 <= hour < 7:
                return 'Early_Morning'
            elif 7 <= hour < 11:
                return 'Morning_Rush'
            elif 11 <= hour < 15:
                return 'Afternoon'
            elif 15 <= hour < 21:
                return 'Afternoon_Rush'
            else:
                return 'Late_Night'
    df['shift'] = df.apply(lambda row: map_hour_to_shift(row['hr'], row['Day']), axis=1)

    # Add quarter column
    def map_month_to_quarter(month):
        if month in ['January', 'February', 'March']:
            return 'Q1'
        elif month in ['April', 'May', 'June']:
            return 'Q2'
        elif month in ['July', 'August', 'September']:
            return 'Q3'
        else:
            return 'Q4'
    df['quarter'] = df['Month'].apply(map_month_to_quarter)

    # Add Daylight column based on month and hour
    def is_daylight(hour, month):
        if month in ['January', 'February', 'March']:
            start_hour, end_hour = 7, 18
        elif month in ['April', 'May', 'June', 'July', 'August']:
            start_hour, end_hour = 6, 20
        elif month in ['September', 'October']:
            start_hour, end_hour = 7, 19
        else:
            start_hour, end_hour = 7, 17
        return 'Yes' if start_hour <= hour < end_hour else 'No'
    df['Daylight'] = df.apply(lambda row: is_daylight(row['hr'], row['Month']), axis=1)

    # Map month and day names to numbers for cyclical features
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    day_map = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }
    df['Monthnum'] = df['Month'].map(month_map)
    df['Daynum'] = df['Day'].map(day_map)

    # Add cyclical features for 'Month', 'Day', and 'hour'
    df['month_sin'] = np.sin(2 * np.pi * df['Monthnum'] / 12).astype(float)
    df['month_cos'] = np.cos(2 * np.pi * df['Monthnum'] / 12).astype(float)
    df['day_sin'] = np.sin(2 * np.pi * df['Daynum'] / 7).astype(float)
    df['day_cos'] = np.cos(2 * np.pi * df['Daynum'] / 7).astype(float)
    df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 24).astype(float)
    df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 24).astype(float)

    # Drop intermediate columns
    df.drop(['Monthnum', 'Daynum'], axis=1, inplace=True)

    # Check if the date is a holiday and display the name
    holiday_name = get_holiday_name(user_date)
    if holiday_name:
        st.write(f"This day is a holiday! The selected day is {holiday_name}")


    # Place the spinner outside the button block to wrap the whole operation
    with st.spinner("Preparing simulation..."):
        # Import model loading function here to ensure it is loaded when this block runs
        from pycaret.regression import load_model
        # Load the model within this block only when the button is clicked
        model = load_model("project/pages/model_f3")

    if st.button("Run Simulation"):
        # Run prediction logic
        prediction = model.predict(df)[0]
        prediction = max(0, prediction)
        st.subheader(f"Predicted bike count at {hour}:00 on {user_date.strftime('%A, %B %d')} is:")
        st.header("**{round(prediction)} bicycles**")



if __name__ == "__main__" or "streamlit" in __name__:
    show_simulation()
