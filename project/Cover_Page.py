import streamlit as st
import importlib.util
from pathlib import Path

# Set up the app layout and default page configuration
st.set_page_config(page_title="Python 2 Final Project", layout="wide")

# Display the main page content with a centered title and delivery date
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Python 2 Final Project - Group 4 </h1>
        <h1>Washington D.C. Bike-Sharing Analysis and Prediction</h1>
        <h3>November 18, 2024</h3>
    </div>
    """,
    unsafe_allow_html=True
)

 # Team Members Section
st.header("Project Team")
    
    # Define team members with specific images and roles
team_data = [
        {"name": "Vitus Schlereth", "role": "Data Engineering Expert", "image": "pages/path_to_vitus.jpg"},
        {"name": "Alina Edigareva", "role": "Data Analytics Expert", "image": "pages/path_to_alina.jpg"},
        {"name": "Yannish Bhandari", "role": "Chief Tech Consultant", "image": "pages/path_to_yannish.jpg"},
        {"name": "Susana Luna", "role": "Data Scientist Expert", "image": "pages/path_to_susana.jpg"},
        {"name": "Gabriel Chapman", "role": "Data Scientist Expert", "image": "pages/path_to_gabriel.jpg"}
    ]
    
    # Display each team member in a larger format
for member in team_data:
    col1, col2 = st.columns([1, 2])  # Adjust column ratio to make images larger
    with col1:
        st.image(member["image"], use_container_width=True)  # Larger image size
    with col2:
        st.subheader(member["name"])
        st.write(member["role"])


# Pages in the side
pages = [
    ("Cover Page", "Main"),
    ("Introduction", "1_Introduction"),
    ("Data Preparation", "2_Data_Preparation"),
    ("EDA Visualization", "3_Data_Visualization"),
    ("Business Insights", "4_Model_Results"),
    ("Model Results", "5_Potential_Improvements"),
    ("Simulation", "6_Simulation")
]


