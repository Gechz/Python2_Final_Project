import streamlit as st

def show_introduction():
    # Center the title using HTML and CSS
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Washington D.C. Bike-Sharing Analysis and Prediction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display images side-by-side using columns and set a consistent aspect ratio
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("project/pages/path_to_capitol.jpg", use_container_width=True)
        
    with col2:
        st.image("project/pages/path_to_bike.jpg", use_container_width=True)

    # Overlay text below the images
    st.write("""
    The Washington D.C. Bike-Sharing Analysis and Prediction project aims to support the administration of Washington D.C. 
    in enhancing the operational efficiency of its bike-sharing service...
    """)

    # Problem Presentation
    st.header("Project Overview")
    st.write("""
    The Washington D.C. Bike-Sharing Analysis and Prediction project aims to support the administration of Washington D.C. 
    in enhancing the operational efficiency of its bike-sharing service. Our consultancy team has been tasked to develop 
    an interactive, data-driven web application that provides both an analytical overview and predictive insights for 
    the city's bike-sharing service. The application will allow the head of transportation services to understand usage 
    patterns, optimize bike provisioning, and potentially reduce operational costs.
    """)
    
    st.subheader("Project Deliverables")
    st.write("""
    This project has two primary deliverables:
    
    1. **Descriptive Analytics**: A comprehensive analysis of bike-sharing usage patterns, including data quality checks, 
       feature engineering, and impactful visualizations. This section aims to give insight into how Washington D.C. 
       citizens utilize the service and highlight areas for potential improvements.
       
    2. **Predictive Model**: A forecasting tool capable of predicting hourly bike usage, enabling the city to anticipate 
       demand more effectively. The predictive model will support informed decisions on bike allocation, ultimately aiming 
       to optimize both service quality and costs.
    """)

# Call the function to display the content when the file is executed directly
if __name__ == "__main__":
    show_introduction()
