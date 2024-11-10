#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#######################
# Page configuration
st.set_page_config(
    page_title="EARFQUAKE Analysis",  # Updated Project Title
    page_icon="🌍",  # Updated with an Earth emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:
    # Sidebar Title
    st.title('EARFQUAKE')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("""
    1. Aguas, Yñikko Arzee Neo
    2. Almandres, Villy Joel
    3. Macabales, Carl Emmanuel
    4. Macatangay, Robin Jairic
    5. Perico, Frederick Lemuel 
    """)

#######################
# Data

# Load data once to avoid redundancy
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

dataset_path = "EARFQUAKE\earthquakes.csv"
try:
    df = load_data(dataset_path)
except FileNotFoundError:
    st.error(f"File not found at path: {dataset_path}")
    st.stop()

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    
    # Adding decorative separator
    st.markdown("---")
    
    # Main introduction with large header and a colorful icon
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Welcome to the Earthquake Data Analysis Tool 🌍</h2>", unsafe_allow_html=True)
    st.write("""
    This app is designed to provide in-depth analysis of earthquake data, enabling users to explore and visualize seismic activity trends, historical data, and key factors that contribute to earthquake occurrence.
    """)

    # Purpose of the Tool with an interactive expander
    with st.expander("🌟 Purpose of the Tool"):
        st.write("""
        The Earthquake Data Analysis Tool was created with the goal of:
        
        - Analyzing historical earthquake data to identify trends and patterns.
        - Helping users visualize seismic activity through interactive charts and maps.
        - Offering insights into factors such as magnitude, depth, and location.
        - Promoting data-driven decision-making in earthquake risk management and preparedness.
        """)

    # How It Works with Tooltips and detailed explanations
    st.subheader("⚙️ How It Works")
    st.markdown("""
    This tool leverages **data science** and **data visualization techniques** to analyze and present earthquake data. Here’s a quick overview of its functionality:
    
    - **Data Input**: Upload earthquake data in CSV format or input manually.
    - **Data Processing**: The app processes the data to extract meaningful patterns and insights.
    - **Visualization**: View detailed visualizations, including charts, graphs, and maps of seismic activity.
    - **Analysis**: Explore data summaries, statistical insights, and key features that influence earthquake behavior.
    """)
    
    st.markdown("""
    <p style='text-align: center;'>Each step allows users to interact with earthquake data and gain valuable insights into seismic activity.</p>
    """, unsafe_allow_html=True)

    # Interactive FAQ section
    st.subheader("❓ Frequently Asked Questions (FAQs)")
    with st.expander("What kind of data can this tool analyze?"):
        st.write("The tool can analyze earthquake data, including information on magnitude, depth, location, time, and more, depending on the dataset provided.")
    with st.expander("Is this tool suitable for real-time analysis?"):
        st.write("Currently, the tool focuses on historical data analysis. Future updates may allow integration with real-time data sources.")
    with st.expander("How accurate are the data visualizations and insights?"):
        st.write("The accuracy depends on the quality of the data provided. We encourage users to upload clean and accurate data for the best insights.")
    
    # Visual Encouragement to Explore
    st.markdown("""
    <div style='text-align: center; color: #4CAF50; font-size: 18px;'>
        🌐 We encourage you to explore the data visualizations and discover valuable insights into seismic activity! 🌐
    </div>
    """, unsafe_allow_html=True)
    
    # Decorative separator
    st.markdown("---")
    
    # Navigation button
    if st.button("🔙 Back to Home"):
        st.session_state.page_selection = "home"


# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Explore the Global Earthquake Data")

    # Introduction with engaging visuals
    st.markdown("""
    The `Global Earthquake Data` dataset, uploaded by **Shreya Sur**, offers a comprehensive view of **1,137 earthquakes** from around the world. It includes important seismic information such as **magnitude**, **depth**, **location**, **time**, and much more. This dataset is perfect for understanding global seismic trends, developing machine learning models, and exploring the real-world impacts of earthquakes.
    """)

    st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Key Features of the Dataset</h3>", unsafe_allow_html=True)

    st.markdown("""
    - 🌍 **Global Coverage**: Earthquakes recorded from various regions across the globe.
    - 📍 **Detailed Location Data**: Including **latitude**, **longitude**, and **country**.
    - 🌊 **Tsunami Information**: Data on tsunami occurrence linked to each earthquake.
    - 🔍 **Seismic Data**: Detailed insights on **magnitude**, **depth**, and **intensity**.
    - 🕒 **Time and Date**: Specific timestamps for each earthquake event.
    """)

    # Add interactivity with an expandable content section
    with st.expander("📋 Dataset Details"):
        st.write("""
        The dataset consists of **1,137 rows** with **43 primary attributes** related to each earthquake event. These attributes provide essential information to study earthquake behavior across different regions.

        ### Key Columns:
        - **Magnitude**: The severity of the earthquake.
        - **Depth**: The depth at which the earthquake occurred.
        - **Latitude** and **Longitude**: Geographic coordinates of the earthquake's epicenter.
        - **Date** and **Time**: When the earthquake occurred.
        - **Type**: The type of earthquake event.
        - **Location**: The specific location where the earthquake occurred.
        - **Continent** and **Country**: Geographical context.
        - **Tsunami Presence**: Whether a tsunami was triggered by the earthquake.

        The data allows for a deep dive into the patterns of seismic activity around the world, helping researchers and enthusiasts alike make sense of the vast earthquake history.
        """)

    st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Dataset Link</h3>", unsafe_allow_html=True)
    st.markdown("[Kaggle - Recent Earthquakes Dataset](https://www.kaggle.com/datasets/shreyasur965/recent-earthquakes/data)")

    st.markdown("---")

    # Interactive Section for Data Preview and Analysis
    st.header("🔍 Dataset Insights")

    st.subheader("👀 Data Preview")
    st.write("Take a quick look at the first few rows of the dataset to understand its structure and contents:")
    st.dataframe(df.head())

    st.subheader("📊 Data Summary")
    st.write("Here’s a statistical overview of the dataset that highlights key numerical features:")
    st.dataframe(df.describe())

    st.subheader("❓ Null Values Check")
    st.write("We can also inspect if there are any missing values in the dataset that might require cleaning or imputation:")
    st.dataframe(df.isnull().sum())

    # Visual enhancements with an interactive map (optional feature, assuming latitude/longitude data exists)
    st.subheader("🌍 Earthquake Locations on Map")
    st.write("Explore the geographical distribution of earthquakes using the interactive map:")
    st.map(df[['latitude', 'longitude']])

    # Data Filtering for Exploration (interactive)
    st.subheader("🛠️ Filter Data")
    st.write("You can filter the dataset based on different criteria like magnitude, depth, or region to dive deeper into the data. Try selecting the parameters below:")

    magnitude = st.slider("Select Magnitude Range", min_value=int(df['magnitude'].min()), max_value=int(df['magnitude'].max()), value=(int(df['magnitude'].min()), int(df['magnitude'].max())))
    filtered_data = df[(df['magnitude'] >= magnitude[0]) & (df['magnitude'] <= magnitude[1])]
    st.dataframe(filtered_data)

    # Encouraging further exploration
    st.markdown("""
    🌐 **Dive deeper** into the dataset by filtering it based on various parameters, exploring the data visually, or analyzing trends over time. Every dataset holds unique insights, and this one is no exception!
    """)

    st.markdown("---")

    # Back to home button
    if st.button("🔙 Back to Home"):
        st.session_state.page_selection = "home"

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    plt.style.use('dark_background')  # Set dark background for all plots

    # Create three columns for better layout
    cols = st.columns((3, 3, 3), gap='medium')

    # Magnitude Distribution
    with cols[0]:
        st.markdown('#### Magnitude Distribution')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.kdeplot(df['magnitude'], shade=True, ax=ax, color='skyblue')
        ax.set_title("Magnitude Distribution (KDE)", color='white')
        ax.set_xlabel("Magnitude", color='white')
        ax.set_ylabel("Density", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Magnitude vs Depth
    with cols[1]:
        st.markdown('#### Magnitude vs Depth')
        fig, ax = plt.subplots(figsize=(8, 5))
        hb = ax.hexbin(df['magnitude'], df['depth'], gridsize=30, cmap="YlGnBu")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Frequency")
        ax.set_title("Magnitude vs Depth Hexbin Plot")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Depth (km)")
        st.pyplot(fig)
        plt.close(fig)


    # Depth Distribution
    with cols[2]:
        st.markdown('#### Depth Distribution')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.violinplot(y=df['depth'], ax=ax, color='lightgreen')
        ax.set_title("Depth Distribution (Violin Plot)", color='white')
        ax.set_ylabel("Depth (km)", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Earthquake Locations
    with cols[0]:
        st.markdown('#### Earthquake Locations')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.kdeplot(x=df['longitude'], y=df['latitude'], cmap="Reds", shade=True, thresh=0.05, ax=ax)
        ax.set_title("Earthquake Location Density Plot", color='white')
        ax.set_xlabel("Longitude", color='white')
        ax.set_ylabel("Latitude", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Earthquake Magnitude by Continent
    with cols[1]:
        st.markdown('#### Magnitude by Continent')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.swarmplot(x='continent', y='magnitude', data=df, ax=ax, palette="viridis")
        ax.set_title("Magnitude Distribution by Continent", color='white')
        ax.set_xlabel("Continent", color='white')
        ax.set_ylabel("Magnitude", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Magnitude by Tsunami Presence
    with cols[2]:
        st.markdown('#### Magnitude by Tsunami Presence')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.histplot(data=df, x='magnitude', hue='tsunami', multiple="stack", bins=30, ax=ax, palette="magma")
        ax.set_title("Magnitude Distribution by Tsunami Presence", color='white')
        ax.set_xlabel("Magnitude", color='white')
        ax.set_ylabel("Frequency", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Hour of Earthquake Occurrence
    with cols[0]:
        st.markdown('#### Hour of Earthquake Occurrence')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['hour'] = df['time'].dt.hour
        hours = df['hour'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))  # Consistent size
        theta = np.linspace(0.0, 2 * np.pi, len(hours), endpoint=False)
        values = hours.values
        bars = ax.bar(theta, values, width=0.3, bottom=0.0, color='teal', alpha=0.7)

        ax.set_xticks(theta)
        ax.set_xticklabels(hours.index, color='white')
        ax.set_title("Earthquake Occurrences by Hour (Polar Plot)", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Magnitude and Depth by Type
    with cols[1]:
        st.markdown('#### Magnitude and Depth by Type')
        pairplot = sns.pairplot(df, vars=['magnitude', 'depth'], hue='type', palette="husl")
        pairplot.fig.suptitle("Magnitude and Depth by Earthquake Type", y=1.02, color='white')
        st.pyplot(pairplot.fig)
        plt.close(pairplot.fig)

    # Magnitude vs Distance from Epicenter
    with cols[2]:
        st.markdown('#### Magnitude vs Distance from Epicenter')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.regplot(x='distanceKM', y='magnitude', data=df, scatter_kws={'alpha': 0.5}, ax=ax, color='orange')
        ax.set_title("Magnitude vs. Distance from Epicenter", color='white')
        ax.set_xlabel("Distance from Epicenter (KM)", color='white')
        ax.set_ylabel("Magnitude", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Insights Section
    st.header("💡 Insights")

    # Re-displaying the graphs without columns
    st.markdown('#### Magnitude Distribution')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(df['magnitude'], shade=True, ax=ax, color='skyblue')
    ax.set_title("Magnitude Distribution (KDE)", color='white')
    ax.set_xlabel("Magnitude", color='white')
    ax.set_ylabel("Density", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: A KDE (Kernel Density Estimate) plot, showing the smooth probability density of earthquake magnitudes. 
             * Interpretation: Shows which earthquake magnitudes are most common. Peaks indicate typical magnitudes, while lower regions reveal rarer magnitudes. Helps to identify the usual intensity range of earthquakes.
             """)

    st.markdown('#### Magnitude vs Depth')
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(df['magnitude'], df['depth'], gridsize=30, cmap="YlGnBu")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Frequency")
    ax.set_title("Magnitude vs Depth Hexbin Plot", color='white')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Depth (km)")
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Hexbin plot showing the distribution of earthquake depth and magnitude.
            * Interpretation: Displays how magnitude changes with depth. Darker hexagons indicate more earthquakes with similar magnitudes and depths, helping to pinpoint common depth levels for high-intensity earthquakes.
            """)

    st.markdown('#### Depth Distribution')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(y=df['depth'], ax=ax, color='lightgreen')
    ax.set_title("Depth Distribution (Violin Plot)", color='white')
    ax.set_ylabel("Depth (km)", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Violin plot showing the distribution of earthquake depths.
             * Interpretation: Highlights the spread of earthquake depths. Wider areas in the plot show the depths where earthquakes are more frequent, helping us understand the most common depth range.
             """)

    st.markdown('#### Earthquake Locations')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(x=df['longitude'], y=df['latitude'], cmap="Reds", shade=True, thresh=0.05, ax=ax)
    ax.set_title("Earthquake Location Density Plot", color='white')
    ax.set_xlabel("Longitude", color='white')
    ax.set_ylabel("Latitude", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Density plot showing areas with the highest concentration of earthquakes on a geographical scale.
             * Interpretation: Highlights the regions with the highest concentration of earthquakes. Brighter areas indicate frequent earthquake activity, identifying regions that may be more seismically active.
             """)

    st.markdown('#### Magnitude by Continent')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.swarmplot(x='continent', y='magnitude', data=df, ax=ax, palette="viridis")
    ax.set_title("Magnitude Distribution by Continent", color='white')
    ax.set_xlabel("Continent", color='white')
    ax.set_ylabel("Magnitude", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Swarm plot showing individual earthquake magnitudes across continents.
             * Interpretation: Shows the magnitude range of earthquakes by continent. Densely packed points indicate frequent magnitudes, helping to identify if certain continents experience stronger earthquakes more often.
             """)

    st.markdown('#### Magnitude by Tsunami Presence')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='magnitude', hue='tsunami', multiple="stack", bins=30, ax=ax, palette="magma")
    ax.set_title("Magnitude Distribution by Tsunami Presence", color='white')
    ax.set_xlabel("Magnitude", color='white')
    ax.set_ylabel("Frequency", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Polar plot showing earthquake occurrences based on the time of day.
             * Interpretation: Compares earthquake magnitudes with tsunami occurrence. Higher bars show the magnitude range where tsunamis are more common, indicating a relationship between higher magnitudes and tsunami generation.
            """)
    
    st.markdown('#### Hour of Earthquake Occurrence')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['hour'] = df['time'].dt.hour
    hours = df['hour'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))
    theta = np.linspace(0.0, 2 * np.pi, len(hours), endpoint=False)
    values = hours.values
    bars = ax.bar(theta, values, width=0.3, bottom=0.0, color='teal', alpha=0.7)

    ax.set_xticks(theta)
    ax.set_xticklabels(hours.index, color='white')
    ax.set_title("Earthquake Occurrences by Hour (Polar Plot)", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""* Graph: Polar plot showing earthquake occurrences based on the time of day.
            * Interpretation: Displays earthquake occurrences by hour in a circular format. Peaks at certain hours indicate the time when earthquakes are more frequent, showing if there are preferred times for earthquake activity.
             """)
    
    st.markdown('#### Magnitude and Depth by Type')
    pairplot = sns.pairplot(df, vars=['magnitude', 'depth'], hue='type', palette="husl")
    pairplot.fig.suptitle("Magnitude and Depth by Earthquake Type", y=1.02, color='white')
    st.pyplot(pairplot.fig)
    plt.close(pairplot.fig)
    st.write("""
             * Graph: Pair plot comparing magnitude, depth, and earthquake type.
             * Interpretation: Compares depth and magnitude across earthquake types. Clusters of points indicate typical depth-magnitude ranges for each type, showing if specific types tend to occur at certain depths or magnitudes.
             """)

    st.markdown('#### Magnitude vs Distance from Epicenter')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x='distanceKM', y='magnitude', data=df, scatter_kws={'alpha': 0.5}, ax=ax, color='orange')
    ax.set_title("Magnitude vs. Distance from Epicenter", color='white')
    ax.set_xlabel("Distance from Epicenter (KM)", color='white')
    ax.set_ylabel("Magnitude", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Scatter plot showing the relationship between magnitude and distance from the epicenter, with a regression line.
             * Interpretation: Shows if earthquake magnitudes change with distance from the epicenter. The regression line reveals any trend—higher or lower magnitudes—relative to their distance from the origin point, indicating whether location affects intensity.
             """)


# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")


# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Add your machine learning code here

# Prediction Page
#######################
# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("📈 Earthquake Prediction")

    # Model Selection
    st.subheader("Model Selection and Configuration")
    model_type = st.selectbox("Choose a Machine Learning Model:", ["Random Forest", "Support Vector Machine (SVM)", "Linear Regression"])
    
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of Estimators:", 10, 200, 100, step=10)
        max_depth = st.slider("Max Depth of Trees:", 1, 20, 10)
    elif model_type == "Support Vector Machine (SVM)":
        kernel = st.selectbox("Choose Kernel:", ["linear", "rbf", "poly"])
        C = st.slider("Regularization Parameter (C):", 0.1, 10.0, 1.0)
    elif model_type == "Linear Regression":
        fit_intercept = st.checkbox("Fit Intercept", value=True)

    # Data Input Options
    st.subheader("Data Input Options")
    data_input_method = st.radio("Choose how to provide data for prediction:", ["Upload Dataset", "Input Manually"])

    if data_input_method == "Upload Dataset":
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
        if uploaded_file:
            prediction_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(prediction_data.head())
    else:
        # Manually enter individual data for a single prediction
        magnitude = st.number_input("Magnitude", min_value=0.0, step=0.1)
        depth = st.number_input("Depth (km)", min_value=0.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.1)
        tsunami = st.selectbox("Tsunami Presence (0 = No, 1 = Yes)", [0, 1])
        
        # Collect data into a DataFrame for a single instance
        prediction_data = pd.DataFrame({
            'magnitude': [magnitude],
            'depth': [depth],
            'latitude': [latitude],
            'longitude': [longitude],
            'tsunami': [tsunami]
        })

    # Button to trigger predictions
    if st.button("Make Predictions"):
        # Placeholder for model training and prediction logic
        # Example code only; replace with actual model and trained weights
        
        # Simulating a prediction
        predictions = np.random.choice(["High Risk", "Medium Risk", "Low Risk"], size=len(prediction_data))

        # Display predictions
        st.write("Prediction Results:")
        prediction_data['Risk Level'] = predictions
        st.dataframe(prediction_data)

        # Feature Importance (if using Random Forest)
        if model_type == "Random Forest":
            st.subheader("Feature Importance")
            feature_importances = np.random.rand(prediction_data.shape[1] - 1)  # Example data
            importance_df = pd.DataFrame({
                'Feature': prediction_data.columns[:-1],
                'Importance': feature_importances
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

        # Visualization of Predictions
        st.subheader("Prediction Visualization")
        fig = px.scatter_mapbox(
            prediction_data,
            lat="latitude",
            lon="longitude",
            color="Risk Level",
            size="magnitude",
            mapbox_style="carto-positron",
            zoom=1,
            title="Earthquake Risk Levels"
        )
        st.plotly_chart(fig)

        # Download Button
        st.subheader("Download Predictions")
        csv_data = prediction_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download as CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")

    else:
        st.info("Provide input data and click 'Make Predictions' to generate risk predictions.")

# Conclusions Page
# Conclusion Page
elif st.session_state.page_selection == "conclusion":
    st.header("🎉 Conclusion and Future Directions")
    
    # Adding a decorative separator
    st.markdown("---")
    
    st.markdown("<h2 style='text-align: center; color: #FF5733;'>🌎 Thank You for Exploring Earthquake Risk Prediction! 🌎</h2>", unsafe_allow_html=True)
    st.write("""
    We hope that this interactive tool provided meaningful insights into earthquake risks. Thank you for using the platform!
    """)

    # Summary of the Tool
    st.subheader("📊 Summary of the Earthquake Prediction Tool")
    st.info("""
    This earthquake prediction tool is designed to help users understand seismic risks with interactive features such as:
    
    - **Model Selection**: Choose between different models to customize predictions.
    - **Input Data Flexibility**: Enter data manually or upload a CSV file.
    - **Visualized Predictions**: View predictions on an interactive map to identify high-risk areas.
    - **Feature Analysis**: Understand which factors influence predictions.
    - **Downloadable Results**: Save your results for further offline analysis.
    
    With this tool, you can make data-driven decisions for earthquake preparedness and risk management.
    """)

    # Insights Section with an Icon
    st.subheader("🔍 Key Insights from Predictions")
    st.markdown("""
    ### What can we learn?
    Through this tool, users can uncover:
    
    - **Impact of Parameters**: Assess how factors like magnitude, depth, and tsunamis affect risk.
    - **Geographic Patterns**: Identify high-risk zones on the interactive map.
    - **Feature Contributions**: Analyze the significance of each parameter on model predictions.
    """)

    # Limitations in an Accordion
    with st.expander("⚠️ Limitations", expanded=True):
        st.write("""
        - **Data Quality**: Prediction accuracy is dependent on training data quality.
        - **Model Simplicity**: Some models may oversimplify earthquake dynamics.
        - **Manual Adjustments Needed**: Users may need to manually adjust inputs for specific data contexts.
        """)
    
    # Future Enhancements with Icons for each bullet
    st.subheader("🚀 Future Enhancements")
    st.markdown("""
    ### How can we improve?
    To make this tool even more powerful, the following enhancements could be considered:

    - **🔧 Model Optimization**: Fine-tune models with hyperparameter tuning techniques.
    - **📊 Enhanced Visualizations**: Integrate 3D maps or heatmaps for better risk visualization.
    - **📡 Real-time Data**: Incorporate live earthquake data to provide up-to-the-minute predictions.
    - **📈 Additional Data Sources**: Use additional geophysical data, like tectonic information, for precision.
    - **🧠 Explainable AI Tools**: Implement SHAP or LIME to help users understand model decisions.
    - **📢 Risk Alerts**: Add a notification feature for high-risk areas based on prediction results.
    """)

    # Divider for visual balance
    st.markdown("---")

    # Thank You Note with Large Font
    st.markdown("<h3 style='text-align: center; color: #FF5733;'>Thank You for Using the Earthquake Prediction Tool! 💖</h3>", unsafe_allow_html=True)
    st.write("""
    We appreciate your interest in earthquake risk analysis. Together, we can work toward better preparedness and resilience in the face of natural disasters. 
    Your feedback is invaluable in enhancing this tool – feel free to reach out with suggestions or ideas!
    """)

    # Encouraging Feedback
    st.markdown("""
    <h4 style='text-align: center;'>Got feedback or suggestions? <br> Let us know how we can improve!</h4>
    """, unsafe_allow_html=True)
    
    # Button for Returning to Home
    if st.button("🏠 Back to Home"):
        st.session_state.page_selection = "home"

# Footer
st.markdown("""
--- 
**Project by Group 3**
""")
