import streamlit as st

def pot_imp():
    # Title 
    st.title("Potential Improvements")

    next_steps = [
        {
            "title": "Scale-Up the Model",
            "text": "This model could be enhanced so it compiles hourly usage per day or week so one can strategize supply accordingly",
            "image_path": "project/pages/path_to_dev.png" 
        },
        {
            "title": "Need for More Data Up to 2024",
            "text": "Expanding the dataset to include more recent data, potentially up to the year 2024, would be beneficial. This would help in capturing more current trends and making future projections more accurate.",
            "image_path": "project/pages/path_to_newdata.jpeg"  
        },
        {
            "title": "Requirement for Additional Variables Such as Zones and Demographic Information",
            "text": "To deepen the analysis and create more segmented insights, it is important to include more variables. Examples include geographical zones, user demographics, and other relevant attributes that could provide a better understanding of different user behaviors.",
            "image_path": "project/pages/path_to_moredata.png" 
        },
        {
            "title": "More Information on Business Values is Needed",
            "text": "There is a need to gather more comprehensive information regarding business-specific values. This could include key performance indicators (KPIs) and metrics related to the financial and operational aspects of the business to connect data findings directly to business outcomes.",
            "image_path": "project/pages/path_to_moreinfo.jpg"  
        },
        {
            "title": "Predict Percentage of Registered Users and Use Synthetic Data",
            "text": "The future course of action involves building predictive models focused on forecasting the percentage of registered users. Additionally, leveraging synthetic data can help expand and enhance the model, providing a more comprehensive understanding and supporting better decision-making.",
            "image_path": "project/pages/path_to_next.png"  
        }
    ]

    # Display Next Steps
    st.header("Elements to Enhance Analysis and Next Steps")
    for i, insight in enumerate(next_steps, 1): 
        col1, col2 = st.columns([3, 1])  
        with col1:
            st.subheader(f"{i}. {insight['title']}")
            st.markdown(insight['text'])
        with col2:
            st.image(insight['image_path'])


if __name__ == "__main__" or "streamlit" in __name__:
    pot_imp()
