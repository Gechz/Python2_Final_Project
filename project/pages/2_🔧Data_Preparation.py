import streamlit as st

def show_data_preparation():
    st.title("Data Preparation")

    # Main Steps section
    with st.expander("Business Front", expanded=True):
        st.subheader("Main Steps")
        st.markdown("""
        - The dataset received did not contain nulls.
        - The dataset provided data points for every hour of every day from 2011 and 2012.
        - General feature engineering was executed to transform data into more processable to enhance visualization and modeling.
        """)

        # Creating Historical Data Features section
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Creating Historical Data Features")
            st.markdown("""
            - Lagged features are created to capture temporal dependencies:
                - **`Previous_Count`**: The bike count from the previous hour.
                - **`Previous_Shift_Mean`**: A 6-hour rolling average of bike counts.
                - **`Previous_Day`**: The bike count from the same hour on the previous day.
            - Rows with missing values due to shifts and rolling means are dropped.
            """)
        with col2:
            st.image("project/pages/path_to_histdata.png")

        # Cyclical Feature Creation section
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cyclical Feature Creation")
            st.markdown("""
            - To handle cyclical data (e.g., `Month`, `Day`, `hr`), sine and cosine transformations are applied:
                - **Month Cyclical Features**: `month_sin`, `month_cos`
                - **Day Cyclical Features**: `day_sin`, `day_cos`
                - **Hour Cyclical Features**: `hour_sin`, `hour_cos`
            - This ensures that the model understands the cyclical nature of these variables (e.g., December is close to January).
            """)
        with col2:
            st.image("project/pages/path_to_cyclicaldata.png")

        # Outcome section
        st.subheader("Outcome")
        st.markdown("""
        - The data is now ready for visualization, exploration, and predictive modeling, considering the other processes found within the technical annex.
        - This prepared dataset includes engineered features that provide deeper insights and help improve model performance.
        """)

    # Other Technical Procedures section
    with st.expander("Technical Annex"):
        st.subheader("Other Technical Procedures")
        st.markdown("""
        - **Datetime Conversion**: The `dteday` column is converted to a `datetime` object to facilitate date-based operations, renamed to `date`, and moved for clarity.
        - **Feature Engineering - Daylight Calculation**:
            - A `Daylight` column indicates whether the time falls within daylight hours, which vary seasonally:
                - **January, February, March**: 7:00 to 18:00
                - **April to August**: 6:00 to 20:00
                - **September, October**: 7:00 to 19:00
                - **November, December**: 7:00 to 17:00
        - **Mapping of Seasons and Weather Conditions**:
            - `season` (1 to 4) is mapped to `spring`, `summer`, `fall`, `winter`.
            - `weathersit` (1 to 4) is mapped to `Clear`, `Misty`, `Light Rain/Snow`, `Heavy Rain/Snow`.
        - **Extracting Date Components**:
            - New columns for `Year`, `Month`, and `Day` extracted from the `date` column for temporal analysis.
        - **Hour-to-Shift Mapping**:
            - The `hr` column is categorized into time shifts:
                - **Weekdays**:
                    - 3:00 to 7:00: Early Morning
                    - 7:00 to 11:00: Morning Rush
                    - 11:00 to 15:00: Afternoon
                    - 15:00 to 21:00: Afternoon Rush
                    - 21:00 to 3:00: Late Night
                - **Weekends**:
                    - 4:00 to 11:00: Early Morning
                    - 11:00 to 18:00: Afternoon
                    - 18:00 to 4:00: Late Night
        - **Quarter Mapping**:
            - `mnth` is mapped to quarters:
                - **Q1**: January to March
                - **Q2**: April to June
                - **Q3**: July to September
                - **Q4**: October to December
        - **Denormalization of Features**:
            - Normalized `temp`, `atemp`, `hum`, and `windspeed` are converted back:
                - **`temp`**: Multiplied by 41
                - **`atemp`**: Multiplied by 50
                - **`hum`**: Multiplied by 100
                - **`windspeed`**: Multiplied by 67
        """)


if __name__ == "__main__" or "streamlit" in __name__:
    show_data_preparation()
