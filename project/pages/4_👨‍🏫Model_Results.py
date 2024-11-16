import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import r2_score

def show_model_results():
    st.title("Model Results")
    @st.cache_data
    def load_predictions():
        # Load the predictions dataset
        return pd.read_csv('project/pages/predictions.csv', parse_dates=['date'])

    # Load the predictions data
    predictions = load_predictions()
    #st.table(predictions.head(2))

    # Model Results placeholder
    with st.expander("Model Results"):
        # Model Top Features
        st.subheader("Most Important Features")
        st.markdown("""
        The most important features contributing to the model's performance, as shown in the Technical Graphs section, are based on the following variables:
        - Historical Data (previous use values for one hour and 24 hour lags)
        - Date Selected
        - Hour Selected
        - Temperature in Celsius
        - Humidity %
        - Windspeed
        """)
         # Model Metrics Summary Table
        st.subheader("Model Performance Metrics")
        
        # DataFrame for model metrics
        metrics_data = {
            "Metric": ["RMSE", "R²", "Adjusted R²", "MAPE"],
            "Train": [23.87, 0.9796, 0.9796, 0.26],
            "Test": [28.68, 0.9830, 0.9829, 0.19]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)

        st.markdown("""
        ### Key Insights
        The model (using a `CatBoost Regressor` estimator) demonstrates strong performance with consistent metrics across both training and testing sets, indicating no signs of overfitting:
        - **RMSE** is 23.87 for the training set and 28.68 for the test set, confirming the model's strong predictive accuracy with minimal deviation between the sets.
        - **R²** values of 0.9796 for training and 0.9830 for testing indicate that the model explains nearly all the variability in the target variable.
        - **Adjusted R²** values of 0.9796 for training and 0.9829 for testing further validate that the model remains highly reliable even when accounting for the number of predictors.
        - **MAPE** values of 0.26 for training and 0.19 for testing suggest that the average percentage error in predictions is minimal, enhancing confidence in forecasting accuracy.

        These results underline the robustness and reliability of the model, making it suitable for real-world applications and strategic decision-making.
        """)

        # Graphs with explanations
        st.subheader("Graphs")
            # Function to plot six-hour aggregation

        # Extract MAPE from the metrics DataFrame
        train_mape = metrics_df.loc[metrics_df['Metric'] == 'MAPE', 'Train'].values[0]
        test_mape = metrics_df.loc[metrics_df['Metric'] == 'MAPE', 'Test'].values[0]

        # Define the subsets for plotting
        first_720 = predictions.head(720)  # First 30 days
        middle_2880 = predictions.iloc[4320:7200]  # Middle 90 days
        last_720 = predictions.tail(720)  # Last 30 days

        # Enhanced function to plot six-hour aggregation with better readability
        def plot_six_hour_aggregation(data, title_suffix, mape_value):
            # Aggregating data for six-hour sum values
            data['six_hour_shift'] = data['date'].dt.floor('6H')
            six_hour_sum = data.groupby('six_hour_shift')['prediction_label'].sum().reset_index()
            six_hour_actual_sum = data.groupby('six_hour_shift')['cnt'].sum().reset_index()

            # Calculating MAPE margins using the extracted MAPE value
            six_hour_sum['upper_mape_margin'] = six_hour_sum['prediction_label'] * (1 + mape_value)
            six_hour_sum['lower_mape_margin'] = six_hour_sum['prediction_label'] * (1 - mape_value)

            # Create a Plotly figure
            fig = go.Figure()

            # Add the six-hour sum prediction line
            fig.add_trace(go.Scatter(
                x=six_hour_sum['six_hour_shift'],
                y=six_hour_sum['prediction_label'],
                mode='lines',
                name='Six-Hour Sum Predicted Count',
                line=dict(color='green', width=2)
            ))

            # Add upper MAPE margin line
            fig.add_trace(go.Scatter(
                x=six_hour_sum['six_hour_shift'],
                y=six_hour_sum['upper_mape_margin'],
                mode='lines',
                name='Upper Bound (MAPE Margin)',
                line=dict(color='orange', dash='dash')
            ))

            # Add lower MAPE margin line
            fig.add_trace(go.Scatter(
                x=six_hour_sum['six_hour_shift'],
                y=six_hour_sum['lower_mape_margin'],
                mode='lines',
                name='Lower Bound (MAPE Margin)',
                line=dict(color='orange', dash='dash')
            ))

            # Add the actual 'cnt' data line (aggregated by six-hour sum)
            fig.add_trace(go.Scatter(
                x=six_hour_actual_sum['six_hour_shift'],
                y=six_hour_actual_sum['cnt'],
                mode='lines',
                name='Six-Hour Sum Actual Count',
                line=dict(color='blue', width=2)
            ))

            # Update layout for better readability
            fig.update_layout(
                title={
                    'text': f'Six-Hour Aggregated Sum Predicted and Actual Counts Over Time with MAPE Margins ({title_suffix})',
                    'font': dict(size=14, color='black'),
                },
                xaxis_title='Date',
                yaxis_title='Sum Bike Count (Six-Hour Period)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    zerolinecolor='black',
                    tickcolor='black',
                    title_font=dict(size=14, color='black'),
                    tickfont=dict(size=12, color='black')
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    zerolinecolor='black',
                    tickcolor='black',
                    title_font=dict(size=14, color='black'),
                    tickfont=dict(size=12, color='black')
                ),
                legend=dict(
                    font=dict(size=12, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.5)'  # Semi-transparent background for the legend
                ),
                paper_bgcolor='white',  # Set paper background color
                plot_bgcolor='white',   # Set plot background color
                margin=dict(l=40, r=40, t=80, b=40)  # Adjust margins for better spacing
            )

            return fig


        # Streamlit selection for which graph to display
        graph_option = st.selectbox(
            "Select the time period for visualization:",
            ["First 30 Days", "Middle 90 Days", "Last 30 Days"]
        )

        # Display the corresponding plot based on the user selection
        if graph_option == "First 30 Days":
            st.plotly_chart(plot_six_hour_aggregation(first_720, "First 30 Days", test_mape))
        elif graph_option == "Middle 90 Days":
            st.plotly_chart(plot_six_hour_aggregation(middle_2880, "Middle 90 Days", test_mape))
        elif graph_option == "Last 30 Days":
            st.plotly_chart(plot_six_hour_aggregation(last_720, "Last 30 Days", test_mape))

        st.subheader("Graph Insights")
        st.markdown("The model seem to fit in line with the actual values, just as expected from our error metrics. These values fit between the MAPE ranges, which is a good sign of forecasting prowess.")

    # Technical Annex
    with st.expander("Technical Annex"):
        # Subsection for Data Preprocessing
        st.subheader("Data Preprocessing for PyCaret")
        st.markdown("""
        This section outlines the key data preprocessing steps applied to the prepared dataset, ensuring it's ready for PyCaret and other machine learning frameworks.
        Below, we explain each step and its significance.
        """)

        st.subheader("1. Outlier Removal Using the IQR Method")
        st.markdown("""
        - Outliers can negatively impact model performance by skewing results or creating biases.
        - The Interquartile Range (IQR) method is used to identify and remove outliers from specific columns:
            - **Columns Considered**: `cnt`, `real_windspeed`, `Previous_Count`, `Previous_Shift_Mean`, `Previous_Day`
        - The formula for outlier detection is:
            - **IQR** = Q3 - Q1
            - Outliers are values outside the range: [Q1 - (multiplier × IQR), Q3 + (multiplier × IQR)]
        - The typical multiplier used is 1.5, but we use 3 for a broader range to reduce unnecessary data loss.
        """)

        st.subheader("2. Data Type Adjustments")
        st.markdown("""
        - To ensure compatibility with PyCaret, specific columns are converted to categorical (object) types:
            - **Columns Converted**:
                - `holiday`
                - `Year`
                - `workingday`
                - `Daylight`
                - `hr`
        - This conversion allows PyCaret to handle these columns as categorical features, ensuring they are treated correctly during model training.
        """)

        st.subheader("3. Feature Selection for PyCaret")
        st.markdown("""
        - Not all columns are used for modeling. Some are dropped because they are not needed or may introduce bias:
            - **Dropped Columns**: `instant`, `casual`, `registered`, `yr`, `mnth`, `temp`, `atemp`, `hum`, `windspeed`, `season`, `weekday`, `weathersit`
        - This step ensures only relevant and meaningful features are used in the model.
        """)

        st.subheader("4. Storing Numerical and Categorical Columns")
        st.markdown("""
        - For further analysis and modeling, it's useful to keep track of the numerical and categorical columns:
            - **Numerical Columns**: Stored in a list for easy reference.
            - **Categorical Columns**: Stored in a separate list.
        - This step simplifies the setup process in PyCaret or any other machine learning framework.
        """)

        st.subheader("Model Selection")
        st.markdown("""
        The Model Selection process is a critical step in developing a reliable and accurate predictive model for bike-sharing demand. 
        This section outlines the configurations used to prepare and select the most suitable model for our analysis.
        
        To achieve optimal results, we utilized the PyCaret library, which offers a comprehensive suite for data preparation, 
        model comparison, and selection. Below, we detail the parameters used in the setup of our modeling environment and their significance.
        """)

        st.subheader("PyCaret Setup Configuration")
        st.markdown("""
        The following configurations were applied during the setup of the PyCaret environment to ensure robust data preprocessing, time series handling, and model training:

        1. **Data Source**:
           - **Dataset**: The final preprocessed dataset (`df_final`), indexed by default to ensure compatibility with PyCaret.
           - **Target Variable**: `cnt`, representing the total count of rental bikes.
           - **Train-Test Split**: An 80/20 split was applied, with 80% of the data used for training and 20% reserved for testing.

        2. **Preprocessing Options**:
           - **Preprocessing Enabled**: Comprehensive preprocessing was applied, including imputation, encoding, and scaling.
           - **Categorical Features**: Specified categorical columns were encoded using One-Hot Encoding for multi-class features and Ordinal Encoding for binary features.
           - **Numeric Features**: Numeric columns were standardized using the z-score method for consistent scaling.

        3. **Feature Normalization**:
           - **Normalization**: Enabled to standardize numeric features.
           - **Method**: Z-score normalization was used to ensure a mean of 0 and a standard deviation of 1.

        4. **Data Transformation**:
           - **Transformation**: Applied to enhance feature distribution for improved model performance.
           - **Method**: Yeo-Johnson transformation was used for handling both positive and negative values effectively.

        5. **Dimensionality Reduction**:
           - **PCA**: Not applied, as dimensionality reduction was not required for this analysis.

        6. **Multicollinearity Handling**:
           - **Remove Multicollinearity**: Enabled to eliminate feature redundancy.
           - **Threshold**: Features with a correlation coefficient above 0.80 were removed to optimize model efficiency.

        7. **Feature Selection**:
           - **Feature Selection**: Enabled to retain the most relevant features.
           - **Method**: The "classic" method (`SelectFromModel` from scikit-learn) was used, keeping the top 50% of features based on importance.

        8. **Time Series Considerations**:
           - **Data Split Strategy**: Time series-specific, ensuring no shuffling (`data_split_shuffle = False`) to maintain the chronological order of observations.
           - **Fold Strategy**: TimeSeriesSplit with 8 folds (`fold_strategy = 'timeseries'`, `fold = 8`), which is essential for time series data to prevent data leakage and maintain temporal integrity during cross-validation.

        9. **Parallel Processing**:
           - **CPU Utilization**: All available CPU cores were used (`n_jobs = -1`) for faster computations.
           - **GPU**: Not utilized in this setup.

        10. **Session Configuration**:
            - **HTML Reporting**: Enabled to generate comprehensive visual reports.
            - **Session ID**: Not specified, resulting in non-reproducible results unless explicitly set.
        """)


        st.subheader("PyCaret Setup Code")
        st.code("""
        from pycaret.regression import setup

        model = setup(
            # Basic Setup
            data = df_final,  # Indexed for visualization purposes, but PyCaret requires a default index
            target = "cnt",
            train_size = 0.8,  # 80% training data

            # Preprocessing
            preprocess = True,
            categorical_features = categorical_columns,  # OHE for multi-class, ordinal for binary categories
            numeric_features = numerical_columns,  # Standardized using z-score method

            # Normalization
            normalize = True,
            normalize_method = 'zscore',

            # Transformation
            transformation = True,
            transformation_method = 'yeo-johnson',  # Proper handling of distribution

            # Dimensionality Reduction
            pca = False,  # Not required

            # Multicollinearity Handling
            remove_multicollinearity = True,
            multicollinearity_threshold = 0.80,  # Correlations above 80% are dropped

            # Feature Selection
            feature_selection = True,
            feature_selection_method = "classic",
            n_features_to_select = 0.50,  # Top 50% selected using SelectFromModel from scikit-learn

            # Time Series Considerations
            data_split_shuffle = False,  # Ensures chronological order
            fold_strategy = 'timeseries',  # Time series-specific cross-validation strategy
            fold = 8,  # 8-fold cross-validation

            # Parallel Processing
            n_jobs = -1,  # Utilize all available cores
            use_gpu = False,  # Not used in this setup

            # Miscellaneous
            html = True,
            session_id = None  # No specific session ID for reproducibility
        )
        """)
        st.write("""
        This setup was designed to ensure a robust training process, optimizing model performance while maintaining interpretability and efficiency, with special attention to time series-specific considerations.
        """)

    # Graphs Expander
    # Technical Graphs
    # Technical Graphs
    # Technical Graphs
    with st.expander("Technical Graphs"):
        st.markdown("""
        This section displays key graphs from the analysis, providing visual insights into the model's performance:
        
        1. **Time Series Plot**: Demonstrates the actual and predicted values over time, allowing you to visually assess the predictive power of the model.
        2. **Residual Plot**: Shows the residuals (differences between actual and predicted values) for visualizing the error distribution over time.
        3. **Predicted vs Actual Scatter Plot**: Compares predicted values against actual values to assess how well the model's predictions align with the real data.
        """)

        # 1. Time Series Plot for Actual vs Predicted Counts
        st.subheader("1. Time Series Plot for Actual vs Predicted Counts")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=predictions['date'], y=predictions['cnt'],
                                  mode='lines', name='Actual Count',
                                  line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=predictions['date'], y=predictions['prediction_label'],
                                  mode='lines', name='Predicted Count',
                                  line=dict(color='red')))

        fig1.update_layout(
            title='Actual vs Predicted Counts Over Time',
            xaxis_title='Date',
            yaxis_title='Bike Count',
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(color='black', size=12),
            title_font=dict(size=18, color='black'),
            xaxis=dict(
                color='black', showgrid=True, gridcolor='lightgray',
                title='Date', title_font=dict(size=14, color='black'), 
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                color='black', showgrid=True, gridcolor='lightgray',
                title='Bike Count', title_font=dict(size=14, color='black'), 
                tickfont=dict(color='black')
            ),
            legend=dict(font=dict(size=12, color='black'))
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**Analysis:** This time series plot helps identify periods where the model aligns closely with actual counts and areas where discrepancies exist. If we refer back to the time series in the data visualization, it follows the same trends")

        # 2. Residual Plot (Residuals = Actual - Predicted)
        st.subheader("2. Residual Plot")
        predictions['residuals'] = predictions['cnt'] - predictions['prediction_label']
        fig2 = px.scatter(predictions, x='date', y='residuals',
                          title='Residual Plot (Actual - Predicted)',
                          labels={'date': 'Date', 'residuals': 'Residuals'})
        fig2.update_traces(marker=dict(color='orange'))
        fig2.update_layout(
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(color='black', size=12),
            title_font=dict(size=18, color='black'),
            xaxis=dict(
                color='black', showgrid=True, gridcolor='lightgray',
                title='Date', title_font=dict(size=14, color='black'), 
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                color='black', showgrid=True, gridcolor='lightgray',
                title='Residuals', title_font=dict(size=14, color='black'), 
                tickfont=dict(color='black')
            ),
            legend=dict(font=dict(size=12, color='black'))
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Analysis :** Residual plots are crucial to check for patterns in prediction errors. Ideally, residuals should be randomly distributed around zero, which they seem to.")

        # 3. Scatter Plot for Predicted vs Actual Counts
        st.subheader("3. Scatter Plot for Predicted vs Actual Counts")
        r2_value = r2_score(predictions['cnt'], predictions['prediction_label'])
        fig3 = px.scatter(predictions, x='cnt', y='prediction_label',
                          title=f'Predicted vs Actual Counts (R²: {r2_value:.2f})',
                          labels={'cnt': 'Actual Count', 'prediction_label': 'Predicted Count'})
        fig3.add_shape(
            type='line', x0=predictions['cnt'].min(), y0=predictions['cnt'].min(),
            x1=predictions['cnt'].max(), y1=predictions['cnt'].max(),
            line=dict(color='green', dash='dash')
        )
        fig3.update_traces(marker=dict(color='purple'))
        fig3.update_layout(
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(color='black', size=12),
            title_font=dict(size=18, color='black'),
            xaxis=dict(
                color='black', showgrid=True, gridcolor='lightgray',
                title='Actual Count', title_font=dict(size=14, color='black'), 
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                color='black', showgrid=True, gridcolor='lightgray',
                title='Predicted Count', title_font=dict(size=14, color='black'), 
                tickfont=dict(color='black')
            ),
            legend=dict(font=dict(size=12, color='black'))
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**Analysis :** The scatter plot helps visualize the overall predictive accuracy. The closer the points align with the diagonal line, the more accurate the predictions.")

# Run the function to display content
if __name__ == "__main__":
    show_model_results()
