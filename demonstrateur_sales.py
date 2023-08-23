import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Loading data
df = pd.read_csv("prevision/sales_data_prevision.csv")
df['Date'] = pd.to_datetime(df['Date'])
df_reel = df[df['Date'] <= '2023-06-30']
df["prediction"] = np.where(df['Date'] == '2021-01-31',800,df["prediction"])

def plot_sales():
    # Tracer la courbe des ventes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_reel['Date'], y=df_reel['TShirt_Sales'],
                             mode='lines', name='TShirt Sales'))

    # Ajouter la courbe des prévisions en rouge
    fig.add_trace(go.Scatter(x=df['Date'], y=df['prediction'],
                             mode='lines', name='Prediction',
                             line=dict(color='red')))

    # Ajouter des labels et un titre
    fig.update_layout(
        title="Sales and Forecasts for TShirt",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Type"
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

# Function to plot stocks and service level for TShirt
def plot_stocks_service_level():
    fig = go.Figure()

    # Tracer le niveau de stock sur l'axe des ordonnées de gauche (yaxis)
    fig.add_trace(go.Scatter(x=df_reel['Date'], y=df_reel['Stock_TShirt'],
                             mode='lines', name='Stock TShirt'))

    # Tracer le niveau de service sur l'axe des ordonnées de droite (yaxis2)
    fig.add_trace(go.Scatter(x=df_reel['Date'], y=df_reel['Service_Level_TShirt'],
                             mode='lines', name='Service Level TShirt', yaxis='y2',line=dict(color='green')))

    # Mise à jour des titres et des labels
    fig.update_layout(
        title="Stock and Service Level for TShirt",
        xaxis_title="Date",
        yaxis_title="Stock TShirt",
        yaxis2=dict(title='Service Level TShirt', overlaying='y', side='right'),
        legend_title="Type"
    )

    st.plotly_chart(fig)

# Function to plot MAE between prediction and actual sales
# Function to plot MAE between prediction and actual sales and Percentage Error
def plot_mae():
    # Compute MAE
    mae = np.abs(df_reel['TShirt_Sales'] - df_reel['prediction'])

    # Compute Percentage Error
    percentage_error = (mae / df_reel['TShirt_Sales']) * 100

    # Plot MAE
    fig = go.Figure(data=go.Scatter(x=df_reel['Date'], y=mae, name='MAE'))

    # Add Percentage Error to the same figure with a secondary y-axis
    fig.add_trace(go.Scatter(x=df_reel['Date'], y=percentage_error, name='Percentage Error', yaxis='y2',line=dict(color="yellow")))

    # Update layout to show two y-axes
    fig.update_layout(
        title="Mean Absolute Error and Percentage Error Between Sales and Prediction",
        xaxis_title="Date",
        yaxis_title="MAE",
        yaxis2=dict(title='Percentage Error (%)', overlaying='y', side='right')
    )

    st.plotly_chart(fig)

st.title("TShirt Sales Forecasting")
# Calculating KPIs

# Filter for 2023 sales data
df_2023 = df[df['Date'].dt.year == 2023]

# Number of T-shirts sold in 2023
tshirts_sold_2023 = df_2023['TShirt_Sales'].sum()

# Number of T-shirts predicted to be sold in 2023
tshirts_predicted_2023 = df_2023['prediction'].sum()

# Calculating stock level for 2023
service_level_2023 = 100*df_2023['Service_Level_TShirt'].mean()  # Suppose we want to display the average stock level for 2023


# Percentage Error
if tshirts_predicted_2023 != 0:
    percentage_error_2023 = ((tshirts_predicted_2023 - tshirts_sold_2023) / tshirts_predicted_2023) * 100
else:
    percentage_error_2023 = 0


# # Load logo image from file
# logo_image = "image\Logo_Iod_solutions_Horizontal_Logo_Complet_Blanc_RVB_1186px@72ppi.png"

# # Display logo in the sidebar
# st.sidebar.image(logo_image, width=100)
st.sidebar.markdown(
"""
<style>
.sidebar .sidebar-content {
    font-family: 'Interstate Regular', Arial, sans-serif;
}
</style>
""",
unsafe_allow_html=True)

st.markdown(
"""
<style>
body {
    font-family: 'Roboto', Arial, sans-serif;
}
</style>
""",
unsafe_allow_html=True)

st.sidebar.markdown("IoD solutions")
analysis_tab = st.sidebar.selectbox('Select a view', ['Overview', 'Analysis'])

if analysis_tab == 'Overview':
    # Displaying KPIs in Streamlit
    st.subheader("Key Performance Indicators for 2023")
    col1, col2, col3, col4 = st.columns(4)
    col1.success(f"**Sold**: {tshirts_sold_2023:,.0f}".replace(',', ' '))
    col2.info(f"**Predicted to be Sold**: {tshirts_predicted_2023:,.0f}".replace(',', ' '))
    col3.warning(f"**Percentage Error**: {percentage_error_2023:.2f}%")
    col4.info(f"**Average Service Level**: {service_level_2023:,.2f}%")  # Adding the stock level KPI

    plot_sales()
    st.subheader('Stock and Service Level for TShirt')
    plot_stocks_service_level() # Combined plot for stock and service level
    st.subheader('Mean Absolute Error for Prediction')
    plot_mae() # Plot for MAE




elif analysis_tab == 'Analysis':

    # Assuming alpha and beta are constants that you've defined or estimated
    alpha = 0.5  # This is just a placeholder value
    beta = 2.0   # This too is just a placeholder value

    # Calculate the impact of a 1% increase in forecast accuracy on service level using the formula
    increase_in_accuracy = 0.05
    impact_on_service_level = beta * np.log(1 + alpha * increase_in_accuracy)

    # Add this impact to the original service level to get the adjusted service level
    original_avg_service_level = df_2023["Service_Level_TShirt"].mean()
    adjusted_avg_service_level = original_avg_service_level + impact_on_service_level

    # Displaying the impact in Streamlit
    st.subheader("Impact of 1% Increase in Forecast Accuracy on Service Level")
    col1, col2, col3 = st.columns(3)
    col1.success(f"Average Service Level for 2023 : {100*original_avg_service_level:.2f}%")
    col2.info(f"Adjusted Average Service Level for 2023 : {100*adjusted_avg_service_level:.2f}%")
    col3.warning(f"Impact on Service Level: {100*impact_on_service_level:.2f}%")


    # 1. Histogram of Errors
    st.subheader("Histogram of Errors")
    errors = df_reel['TShirt_Sales'] - df_reel['prediction']
    fig = go.Figure(data=[go.Histogram(x=errors)])
    st.plotly_chart(fig)

    # 2. Seasonality using a Simple Moving Average
    st.subheader("Seasonality: Moving Average")
    df_reel['SMA'] = df_reel['TShirt_Sales'].rolling(window=3).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_reel['Date'], y=df_reel['TShirt_Sales'], name="Actual Sales"))
    fig.add_trace(go.Scatter(x=df_reel['Date'], y=df_reel['SMA'], name="Moving Average", line=dict(color='red')))
    st.plotly_chart(fig)