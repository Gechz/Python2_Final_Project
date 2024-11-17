import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("project/pages/hour.csv")

# Transform the data as needed for EDA
@st.cache_data
def transform_data(df):
    # Convert 'dteday' to datetime format
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['date'] = pd.to_datetime(df['dteday'])
    # Add Year, Month, Day, and weekday columns
    df['Year'] = df['dteday'].dt.year
    df['Month'] = df['dteday'].dt.month
    df['Day'] = df['dteday'].dt.day
    df['weekday'] = df['dteday'].dt.day_name()

    # Map seasons and add a column for readable season names
    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season_name'] = df['season'].map(season_mapping)

    # Classify days as working or non-working
    df['workingday_name'] = df['workingday'].apply(lambda x: 'Working Day' if x == 1 else 'Non-Working Day')

    # Add cyclical features for hour, month, and weekday for better visualization
    df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 24).astype(float)
    df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 24).astype(float)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12).astype(float)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12).astype(float)
    df['day_sin'] = np.sin(2 * np.pi * df['dteday'].dt.dayofweek / 7).astype(float)
    df['day_cos'] = np.cos(2 * np.pi * df['dteday'].dt.dayofweek / 7).astype(float)

    # Add quarter and shift columns
    df['quarter'] = df['Month'].apply(lambda x: 'Q1' if x <= 3 else ('Q2' if x <= 6 else ('Q3' if x <= 9 else 'Q4')))
    df['shift'] = df.apply(lambda row: map_hour_to_shift(row['hr'], row['weekday']), axis=1)

    # Add daylight column based on month and hour
    df['Daylight'] = df.apply(lambda row: is_daylight(row['hr'], row['Month']), axis=1)

    # Add previous hour, shift mean, and previous day bike counts
    df['Previous_Count'] = df['cnt'].shift(1)  # Previous hour
    df['Previous_Shift_Mean'] = df['cnt'].rolling(6).mean()  # Average of the last 6 hours
    df['Previous_Day'] = df['cnt'].shift(24)  # Previous day

    # Drop rows with nulls created by shift and rolling operations
    df = df.dropna(axis=0)

    # Drop the 'instant' column as it behaves like an index
    df = df.drop(columns=["instant"])

    # Add real values for temperature, atemp, hum, and windspeed
    temp_max = 41
    atemp_max = 50
    hum_max = 100
    windspeed_max = 67
    df['real_temp'] = round(df['temp'] * temp_max, 1)
    df['real_atemp'] = round(df['atemp'] * atemp_max, 1)
    df['real_hum'] = round(df['hum'] * hum_max, 1)
    df['real_windspeed'] = round(df['windspeed'] * windspeed_max, 1)

    return df

# Helper functions for mapping shift and determining daylight
def map_hour_to_shift(hour, weekday):
    if weekday in ['Saturday', 'Sunday']:
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
def is_daylight(hour, month):
    if month in [1, 2, 3]:
        start_hour, end_hour = 7, 18
    elif month in [4, 5, 6, 7, 8]:
        start_hour, end_hour = 6, 20
    elif month in [9, 10]:
        start_hour, end_hour = 7, 19
    else:
        start_hour, end_hour = 7, 17
    return 'Yes' if start_hour <= hour < end_hour else 'No'

# Function to create and display the time series plot
def display_time_series(df):
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the weekly sum of 'cnt' for 7-day intervals
    weekly_total_usage = df.set_index('date').resample('7D')['cnt'].sum().reset_index()
    weekly_total_usage.columns = ['date', 'total_weekly_usage']  # Rename columns for clarity

    # Create the time series line chart with vertical lines at each weekly interval
    fig = go.Figure()

    # Add main line for total weekly usage
    fig.add_trace(go.Scatter(
        x=weekly_total_usage['date'], 
        y=weekly_total_usage['total_weekly_usage'], 
        mode='lines+markers',
        line=dict(color='#800020', width=2),  # Burgundy red line for total weekly usage
        marker=dict(size=6),  # Markers to highlight each data point
        name='Total Weekly Usage'
    ))

    # Add vertical lines for each week
    for i, row in weekly_total_usage.iterrows():
        fig.add_shape(
            type="line",
            x0=row['date'], y0=0,
            x1=row['date'], y1=row['total_weekly_usage'],
            line=dict(color="gray", width=1, dash="dot")
        )

    # Customize layout for better readability
    fig.update_layout(
        title="Total Weekly Usage Over Time with Weekly Intervals",
        xaxis_title="Date",
        yaxis_title="Total Usage Count",
        paper_bgcolor="white",  # White background for better readability
        plot_bgcolor="white",   # White plot background for better readability
        font=dict(color="black"),  # Black font for contrast
        title_font=dict(size=20, color="black"),
        legend=dict(
            title="Legend",
            font=dict(color="black"),
            bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent white background for the legend
            bordercolor="black",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            linecolor="black",  # Black axis lines for better contrast
            tickfont=dict(color="black"),  # Black font for x-axis ticks
            title_font=dict(size=14, color="black")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            linecolor="black",  # Black axis lines for better contrast
            zeroline=True,
            zerolinecolor="black",
            tickfont=dict(color="black"),  # Black font for y-axis ticks
            title_font=dict(size=14, color="black")
        )
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)


# Function to create and display user type behavior plots
def display_user_type_behavior(df, selected_years, selected_user_types, time_unit):
    # Create temporary columns for registered and casual percentages
    df['registered_percentage'] = (df['registered'] / df['cnt']) * 100
    df['casual_percentage'] = (df['casual'] / df['cnt']) * 100

    # Separate data by year
    df_2011 = df[df['Year'] == 2011]
    df_2012 = df[df['Year'] == 2012]

    # Map user-friendly names to actual DataFrame columns
    time_unit_mapping = {
        "Hour": 'hr',
        "Day of the Week": 'weekday',
        "Month": 'Month',
        "Season": 'season_name',
        "Quarter": 'quarter'
    }

    # Check if the selected time unit exists in the mapping
    if time_unit not in time_unit_mapping:
        st.error("Invalid time unit selected.")
        return

    column_for_grouping = time_unit_mapping[time_unit]

    # Ensure weekdays are sorted in the correct order if the time unit is 'Day of the Week'
    if time_unit == "Day of the Week":
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=day_order, ordered=True)

    # Calculate the mean registered and casual percentages by each time unit
    def calculate_percentage(df, column, percentage_col):
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        grouped_data = df.groupby(column)[percentage_col].mean()
        grouped_data = grouped_data.reindex(day_order) if column == 'weekday' else grouped_data  # Ensure order for weekdays
        return grouped_data

    # Initialize figure
    fig = go.Figure()

    # Add traces based on user selections
    colors = {'2011': '#004225', '2012': '#800020'}  # Green for 2011 and burgundy for 2012
    line_styles = {'Registered': 'solid', 'Casual': 'dot'}

    for year in selected_years:
        year_data = df_2011 if year == 2011 else df_2012
        for user_type in selected_user_types:
            column = 'registered_percentage' if user_type == "Registered" else 'casual_percentage'
            mean_data = calculate_percentage(year_data, column_for_grouping, column)
            fig.add_trace(go.Scatter(
                x=mean_data.index,
                y=mean_data.values,
                mode='lines+markers',
                name=f'{year} {user_type}',
                line=dict(color=colors[str(year)], width=2, dash=line_styles[user_type]),
                marker=dict(size=6, symbol='circle')
            ))

    # Update layout to make axis titles and tick labels black and tick marks black
    fig.update_layout(
        title=f"{' and '.join(selected_user_types)} Users Percentage by {time_unit}",
        xaxis_title=time_unit,
        yaxis_title="Percentage (%)",
        yaxis=dict(
            range=[0, 100],
            title_font=dict(color="black"),  # Black y-axis title
            tickfont=dict(color="black"),  # Black y-axis tick labels
            tickcolor="black",  # Black tick marks
        ),
        xaxis=dict(
            title_font=dict(color="black"),  # Black x-axis title
            tickfont=dict(color="black"),  # Black x-axis tick labels
            tickcolor="black",  # Black tick marks
            showgrid=True,
            gridcolor="lightgray",
            categoryorder='array',  # Ensure the correct order for days of the week
            categoryarray=day_order if time_unit == "Day of the Week" else None
        ),
        paper_bgcolor="#ffffff",  # White background for better contrast
        plot_bgcolor="#ffffff",
        font=dict(color="black"),  # Ensures other text elements are black
        title_font=dict(size=20, color="black"),
        legend=dict(font=dict(color="black")),
    )

    # Drop the temporary columns after plotting
    df.drop(columns=['registered_percentage', 'casual_percentage'], inplace=True)

    # Display the figure in Streamlit
    st.plotly_chart(fig)

# Function to create and display heatmaps
def display_heatmaps(df, selected_y_axis, selected_heatmaps):
    # Map weekday labels to 'Weekday' or 'Weekend'
    df['weekday_label'] = df['weekday'].apply(lambda x: 'Weekday' if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 'Weekend')
    
    # Map month numbers to month names and ensure correct order
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    df['month_name'] = pd.Categorical(df['Month'].map(month_map), categories=list(month_map.values()), ordered=True)

    # Map month numbers to quarters
    df['quarter'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], labels=["Q1", "Q2", "Q3", "Q4"], right=True)

    # Create a pivot table for the selected y-axis feature
    def create_pivot(data, value, y_axis_feature):
        if y_axis_feature == "Weekday/Weekend":
            return data.groupby(['weekday_label', 'hr'])[value].mean().unstack()
        elif y_axis_feature == "Month":
            return data.groupby(['month_name', 'hr'])[value].mean().unstack()
        elif y_axis_feature == "Season":
            return data.groupby(['season_name', 'hr'])[value].mean().unstack()
        elif y_axis_feature == "Quarter":
            return data.groupby(['quarter', 'hr'])[value].mean().unstack()

    # Create pivot tables for cnt, registered, and casual
    pivot_tables = {
        "Total Count": create_pivot(df, 'cnt', selected_y_axis),
        "Registered Users": create_pivot(df, 'registered', selected_y_axis),
        "Casual Users": create_pivot(df, 'casual', selected_y_axis)
    }

    # Function to create a heatmap figure
    def create_heatmap(data, title, color_scale):
        fig = px.imshow(
            data,
            labels=dict(x="Hour of the Day", y=selected_y_axis, color="Average Count"),
            color_continuous_scale=color_scale,
            title=title,
        )
        fig.update_layout(
            width = 500,
            height = 350,
            xaxis=dict(tickmode='linear', color='black', title="Hour of the Day", title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(color='black', title=selected_y_axis, title_font=dict(color='black'), tickfont=dict(color='black')),
            title_font=dict(color='black'),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            coloraxis_colorbar=dict(tickcolor='black', title_font_color='black', tickfont_color='black')
        )
        return fig

    # Display selected heatmaps with unique keys
    for heatmap in selected_heatmaps:
        if heatmap in pivot_tables:
            color_scale = "Burg" if heatmap == "Total Count" else "Greens" if heatmap == "Registered Users" else "Blues"
            st.plotly_chart(create_heatmap(pivot_tables[heatmap], f"Average {heatmap} by {selected_y_axis} and Hour", color_scale), use_container_width=True, key=f"{heatmap}_{selected_y_axis}")

# Function to create and display scatter plots for meteorological effects
def display_scatter_plots(df, selected_user_type, selected_meteorological_factors):
    # Map user type to column names
    user_type_mapping = {
        "Total Count": "cnt",
        "Registered Users": "registered",
        "Casual Users": "casual"
    }
    user_type_column = user_type_mapping[selected_user_type]

    # Map meteorological factors to column names
    meteorological_mapping = {
        "Real Temperature": "real_temp",
        "Perceived Temperature": "real_atemp",
        "Real Humidity": "real_hum",
        "Real Windspeed": "real_windspeed"
    }

    # Loop through each selected meteorological factor and create individual scatter plots
    for factor in selected_meteorological_factors:
        meteorological_column = meteorological_mapping[factor]

        # Create scatter plot
        fig = px.scatter(
            df,
            x=meteorological_column,
            y=user_type_column,
            labels={meteorological_column: factor, user_type_column: f"{selected_user_type} Count"},
            title=f"{selected_user_type} vs {factor}",
            color=user_type_column,
            color_continuous_scale="Viridis"
        )

        # Update layout for better visibility
        fig.update_layout(
            xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
            title_font=dict(color='black'),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            coloraxis_colorbar=dict(tickcolor='black', title_font_color='black', tickfont_color='black')
        )

        # Display scatter plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

# Function to create and display histogram and boxplot
def display_histogram_and_boxplot(df,selected_label):

    # Create a figure with two subplots (histogram and boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Mapping dictionary for user-friendly labels
    feature_labels = {
    'Count': 'cnt',
    'Registered': 'registered',
    'Casual': 'casual',
    'Temperature': 'temp',
    'Perceived Temperature': 'atemp',
    'Humidity': 'hum',
    'Windspeed': 'windspeed'
    }
    # Get the internal column name for the selected label
    selected_feature = feature_labels[selected_label]
    # Histogram
    sns.histplot(df[selected_feature], kde=True, color="orange", ax=axes[0])
    axes[0].set_title(f'Histogram of {selected_label.capitalize()}', color='black', fontsize=16)
    axes[0].set_xlabel(selected_feature.capitalize(), color='black', fontsize=12)
    axes[0].set_ylabel('Frequency', color='black', fontsize=12)
    axes[0].set_facecolor('white')

    # Boxplot
    sns.boxplot(
        x=df[selected_feature],
        color="orange",
        flierprops=dict(marker='o', markerfacecolor='orange', markersize=5),
        ax=axes[1]
    )
    axes[1].set_title(f'Boxplot of {selected_label.capitalize()}', color='black', fontsize=16)
    axes[1].set_xlabel(selected_feature.capitalize(), color='black', fontsize=12)
    axes[1].set_facecolor('white')

    # Set overall figure background color
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Function to create and display the correlation matrix
def display_correlation_matrix(df,filter_correlation):
    # Filter to include only numeric columns
    numeric_df = df.select_dtypes(include=[float, int])

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()

    # Apply a mask if filtering is selected
    if filter_correlation:
        mask = corr_matrix.abs() < 0.5
    else:
        mask = None

    # Set up the figure
    plt.figure(figsize=(16, 14))
    plt.gcf().patch.set_facecolor('#0a0f17')  # Set figure background color

    # Define a diverging color map
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw the heatmap
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"shrink": .75, "label": "Correlation Coefficient"},
        annot_kws={"size": 10, "color": "black"},
        square=True
    )

    # Set the white background for the heatmap itself
    ax.set_facecolor("white")

    # Title and axis customization
    plt.title("Correlation Matrix", color="white" if filter_correlation else "#e6eaf0", fontsize=16)
    plt.xticks(color="white", rotation=45, ha='right', fontsize=10)
    plt.yticks(color="white", fontsize=10)

    # Customize color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")

    # Display the plot in Streamlit
    st.pyplot(plt)

# Load and transform data
df = load_data()
df = transform_data(df)

# Display a preview of the data
st.subheader("Brief Dataframe inspection after transformations:")
st.dataframe(df.iloc[:,-18:].head(3))

feature_labels = {
'Count': 'cnt',
'Registered': 'registered',
'Casual': 'casual',
'Temperature': 'temp',
'Perceived Temperature': 'atemp',
'Humidity': 'hum',
'Windspeed': 'windspeed'
}

# Function to display the EDA page
def show_eda():
    st.title("Data Visualization")


    # Placeholder for visualization content
    st.markdown("""
    Explore various visualizations below such as:
    - Histograms for bike counts
    - Time-series analysis of bike usage trends
    - Boxplots for bike demand by season
    - Scatter plots for temperature vs. bike count
    """)

    # Business Insights section
    with st.expander("Business Insights"):
        # Time Series -- Finding Trends over Time
        st.header("1. Time Series -- Finding Trends over Time")
        display_time_series(df)
        st.subheader("Time Series Insight")
        st.markdown(
            """
            **Analysis**:
            - The number of total users utilizing the service has shown a consistent increase over time, indicating growth in overall popularity and user engagement with the platform. The dip at the end could be because of change of service, either way it looks unnatural.
            """
        )

        # User Type Behavior over time
        st.header("2. User Type Behavior over Time")
        st.sidebar.header("User Type Behavior Options")
        # User selections for user type behavior visualization
        selected_years = st.sidebar.multiselect("Select Year(s):", [2011, 2012], default=[2011, 2012])
        selected_user_types = st.sidebar.multiselect("Select User Type(s):", ["Registered", "Casual"], default=["Registered"])
        time_unit = st.sidebar.selectbox("Select Time Granularity:", ["Month","Day of the Week", "Hour", "Season", "Quarter"])
        # Call the display_user_type_behavior function with user selections
        display_user_type_behavior(df, selected_years, selected_user_types, time_unit)
        st.subheader("User Type Behavior Insight")
        st.markdown(
            """
            **Analysis**:
            - This visualization highlights that the proportion of registered users has been increasing ever so slightly over time, suggesting a trend towards more frequent and committed usage. 
            - As we saw in the previous graph, there was a massive turndown in overall usage at December 2012, so if we focus at middle of the year months, this trend is clearer.
            """
        )

        # Heatmaps -- Identifying Peak Usage
        st.header("3. Heatmaps -- Identifying Peak Usage")
        st.sidebar.header("Heatmap Options")
        # User selections for heatmaps
        selected_y_axis = st.sidebar.selectbox("Select Y-axis Feature:", ["Month","Weekday/Weekend", "Season", "Quarter"])
        selected_heatmaps = st.sidebar.multiselect(
            "Select Heatmaps to Display:", ["Total Count", "Registered Users", "Casual Users"], default=["Registered Users", "Casual Users"]
        )
        # Call the display_heatmaps function with user selections
        display_heatmaps(df, selected_y_axis, selected_heatmaps)
        st.subheader("Heatmap Insight")
        st.markdown(
            """
            **Analysis**:
            - Different types of users, such as casual and registered users, display distinct usage peaks. Recognizing these variations can assist in strategic resource scheduling and service optimization.
            - Investigating consistent trends without significant changes can help confirm seasonality, which is crucial for understanding predictable fluctuations and making informed operational decisions.
            """
        )

        # Scatter Plots -- Meteorological Effects
        st.header("4. Scatter Plots -- Meteorological Effects")
        st.sidebar.header("Scatter Plot Options")
        selected_user_type = st.sidebar.selectbox("Select User Type:", ["Total Count", "Registered Users", "Casual Users"])
        selected_meteorological_factors = st.sidebar.multiselect(
            "Select Meteorological Factors:", ["Real Temperature", "Perceived Temperature", "Real Humidity", "Real Windspeed"],
            default=["Real Temperature"]
        )
        display_scatter_plots(df, selected_user_type, selected_meteorological_factors)
        st.subheader("Scatter Plot Insight")
        st.markdown(
            """
            **Analysis**:
            - Weather conditions, such as temperature, humidity, and other meteorological factors,  influence user behavior and service usage patterns. Overall, real temperature seems to be the one with highest influence. Integrating weather-related data into analysis can provide more comprehensive insights.
            """
        )

    # Technical Visualizations section (Placeholder)
    with st.expander("Technical Visualizations"):

         # Histogram and Boxplot of Features
        st.header("1. Histogram and Boxplot of Features")
        # Sidebar header and feature selection with user-friendly labels
        st.sidebar.header("Histogram and Boxplot Options")
        selected_label = st.sidebar.selectbox("Select a Feature to Inspect:",options=list(feature_labels.keys()))  
        display_histogram_and_boxplot(df,selected_label)
        st.markdown("**Analysis Placeholder:** Histograms and boxplots help in understanding the distribution and identifying outliers in the data.")

        # Correlation Matrix
        st.header("2. Correlation Matrix")
        st.sidebar.header("Correlation Matrix Options")
        filter_correlation = st.sidebar.checkbox("Show Only Correlations >= |0.5|")
        display_correlation_matrix(df,filter_correlation)
        st.markdown("**Analysis Placeholder:** The correlation matrix highlights relationships between numerical features. Filtering can help focus on stronger correlations.")


# Run the function to display content
if __name__ == "__main__" or "streamlit" in __name__:
    show_eda()
