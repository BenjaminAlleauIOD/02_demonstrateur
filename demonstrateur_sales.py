import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

def generate_fake_data():
    num_products = 50
    products = ['Product ' + str(i) for i in range(1, num_products+1)]
    sales_forecast = np.random.randint(500, 1000, num_products)
    sales_actuals = np.random.randint(400, 1000, num_products)
    forecast_error = sales_actuals - sales_forecast
    current_date = datetime.now()
    date_range = pd.date_range(end=current_date, periods=num_products)
    start_date = date_range.min()
    end_date = date_range.max()

    data = pd.DataFrame({'Product': products,
                         'Sales Forecast': sales_forecast,
                         'Sales Actuals': sales_actuals,
                         'Forecast Error': forecast_error,
                         'Date': date_range})

    return data, start_date, end_date

data, start_date, end_date = generate_fake_data()

if __name__ == '__main__':
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

    st.sidebar.title("")
    tab_selected = st.sidebar.radio("Navigation", options=["Overview", "Sales Forecast"])

    selected_data = data

    st.title("Sales Forecast Dashboard 2023")

    if tab_selected == "Overview":

        st.subheader("General Information")

        col1, col2, col3 = st.columns(3)
        num_products = len(selected_data)
        col1.info(f"Number of Products : **{num_products}**")

        total_forecast = selected_data['Sales Forecast'].sum()
        total_actuals = selected_data['Sales Actuals'].sum()
        total_error = total_actuals - total_forecast

        col2.info(f"Total Sales Forecast: **{total_forecast}**")
        col3.info(f"Total Sales Actuals: **{total_actuals}**")

        fig = px.line(data, x='Date', y='Sales Forecast', title="Sales Forecast over Time")
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Sales Forecast")

        st.plotly_chart(fig)

    elif tab_selected == "Sales Forecast":
        st.subheader("Sales Forecast")

        product_name = st.selectbox("Select a Product", selected_data['Product'].unique())

        if product_name:
            product_info = selected_data[selected_data['Product'] == product_name].iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.subheader("Sales Forecast")
            col1.info(product_info['Sales Forecast'])
            col2.subheader("Sales Actuals")
            col2.info(product_info['Sales Actuals'])
            col3.subheader("Forecast Error")
            col3.info(product_info['Forecast Error'])
