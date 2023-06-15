import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import networkx as nx

st.set_option('deprecation.showPyplotGlobalUse', False)
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import got 


## Function
def generate_fake_data():
    num_suppliers = 50
    suppliers = ['F ' + str(i) for i in range(1, num_suppliers+1)]
    addresses = ['Address ' + str(i) for i in range(1, num_suppliers+1)]
    delivery_times = np.random.randint(70, 100, num_suppliers)
    risk_scores = np.random.randint(1, 100, num_suppliers)
    revenue = np.random.randint(1, 100, num_suppliers)
    compliance_rate_ind = np.random.uniform(0.5, 1.0, num_suppliers)
    risk_factors = np.random.choice(['high price variation', 'increased number of delays', 'increased number of defects'], num_suppliers)
    compliance_rate = np.random.uniform(0.5, 1.0)
    num_delays = np.random.randint(0, 4, num_suppliers)
    risk_variation = np.random.randint(-10, 30, num_suppliers)
    current_date = datetime.now()
    date_range = pd.date_range(end=current_date, periods=num_suppliers)
    start_date = date_range.min()
    end_date = date_range.max()

    data = pd.DataFrame({'Supplier': suppliers,
                         'Address': addresses,
                         'Delivery Time': delivery_times,
                         'Risk Score': risk_scores,
                         'Revenue': revenue,
                         'Risk Factor': risk_factors,
                         'Risk Variation' : risk_variation,
                         'Non-compliance Rate': compliance_rate_ind,
                         'Number of Delays': num_delays,
                         'Date': date_range})

    # Generate delay data
    months = pd.date_range(start='2022-01-01', periods=12, freq='M')
    delay_counts = np.random.randint(low=20, high=100, size=12)
    df_delays = pd.DataFrame({'Month': months, 'Delay Count': delay_counts})

    # Generate delivery time data
    months = pd.date_range(start='2022-01-01', periods=12, freq='M')
    delivery_time = np.random.randint(low=20, high=100, size=12)
    n1 = np.random.randint(low=50, high=100, size=12)
    df_delivery = pd.DataFrame({'Month': months, 'Delivery Time': delivery_time,"n1":n1})
    


    return data, compliance_rate, start_date, end_date, df_delays, df_delivery

## Generate fake data
data, compliance_rate, start_date, end_date, df_delays, df_delivery = generate_fake_data()

## generate fake date graph 
# Génération de données factices pour les fournisseurs
num_suppliers = 6
suppliers = ['F ' + str(i) for i in range(1, num_suppliers+1)]
connections = np.random.randint(0, 2, size=(num_suppliers, num_suppliers))

# Création du DataFrame des connexions
df_connections = pd.DataFrame(connections, index=suppliers, columns=suppliers)

# Création du graphe à partir du DataFrame
graph = nx.from_pandas_adjacency(df_connections)

## generer feature importance fictive
# Génération de valeurs fictives pour l'importance des facteurs
risk_factors = ['high price variation', 'increased number of delays', 'increased number of defects']

feature_importance = np.random.uniform(0, 1, len(risk_factors))
feature_importance = feature_importance / feature_importance.sum()  # Normalisation des valeurs pour qu'elles se situent entre 0 et 1

# Création du DataFrame des feature importance
df_feature_importance = pd.DataFrame({'Risk Factors': risk_factors, 'Importance': feature_importance})

# Tri des facteurs par ordre décroissant d'importance
df_feature_importance = df_feature_importance.sort_values('Importance', ascending=False)



# Application Execution
if __name__ == '__main__':

    # Load logo image from file
    logo_image = "image\Logo_Iod_solutions_Horizontal_Logo_Complet_Blanc_RVB_1186px@72ppi.png"

    # Display logo in the sidebar
    st.sidebar.image(logo_image, width=100)

    st.sidebar.title("")
    tab_selected = st.sidebar.radio("Navigation", options=["Overview", "Suppliers","Graph"])

    # Filter data based on selected period
    selected_data = data  # Display all data

    # Streamlit Application Configuration
    st.title("Supplier Dashboard 2023")

    if tab_selected == "Overview":

        # Display general information
        st.subheader("General Information")

        # Number of suppliers
        col1, col2, col3 = st.columns(3)
        num_suppliers = len(selected_data)
        col1.info(f"Number of suppliers : **{num_suppliers}**")

        # Average delivery time
        col2.success(f"Delivery on Time : **60%**")

        # Number of delayed deliveries
        col3.warning(f"Non-Default Delivery : **85%**")

        # Top 5 high-risk suppliers
        col1, col2 = st.columns(2)
        col1.subheader("High-Risk Suppliers")
        #top_risk_suppliers = selected_data.nlargest(5, 'Risk Score')
        col1.dataframe(selected_data[["Supplier", "Risk Score", "Risk Variation"]].sort_values("Risk Variation",ascending=False), hide_index=True)

        # Création du graphique
        # Création de la figure Plotly

        fig = go.Figure(go.Bar(
            x=risk_factors,
            y=feature_importance,
            marker={'color': 'rgb(65,105,225)'}
        ))

        # Mise en forme de la figure
        fig.update_layout(
            xaxis_title="Risk Factors",
            yaxis_title="Importance",
        )

        # Affichage de la figure dans Streamlit
        col2.subheader("Feature Importance - Risk Factors")
        col2.plotly_chart(fig, use_container_width=True)
        # # Compliance Rate
        # with col2:
        #     gauge = go.Figure(go.Indicator(
        #         domain={'x': [0, 1], 'y': [0, 1]},
        #         value=round(100 * compliance_rate),
        #         mode="gauge+number+delta",
        #         title={'text': "Compliance Rate"},
        #         delta={'reference': 80},
        #         gauge={'axis': {'range': [None, 100]},
        #                'steps': [
        #                    {'range': [0, 50], 'color': "lightgray"},
        #                    {'range': [50, 100], 'color': "gray"}],
        #                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 65}}))
        #     gauge = gauge.update_layout(autosize=True)

        #     st.plotly_chart(gauge, theme="streamlit", use_container_width=True)

        # Delay Count Trend
        st.subheader("Delivery Trend Global")
        # Create interactive chart with Plotly
        fig = px.line(df_delivery, x='Month', y='Delivery Time', title="Delivery On-Time Trend (D1) and Non-Default Delivery Trend per Month (N1)")
        fig.update_xaxes(title="Month")
        fig.update_yaxes(title="%")
        fig.add_trace(go.Scatter(x=df_delivery['Month'], y=df_delivery['Delivery Time'], mode='lines', name='D1'))
        fig.add_trace(go.Scatter(x=df_delivery['Month'], y=df_delivery['n1'], mode='lines', name='N1'))

        # Display interactive chart with Streamlit
        st.plotly_chart(fig)

    elif tab_selected == "Suppliers":
        st.subheader("Suppliers")

        # Select a supplier
        supplier_name = st.selectbox("Select a Supplier", selected_data['Supplier'].unique())

        if supplier_name:
            supplier_info = selected_data[selected_data['Supplier'] == supplier_name].iloc[0]
            col1, col2, col3 = st.columns(3)
            # Address
            col1.subheader("Address")
            col1.info(supplier_info['Address'],icon="🏘️")

            # Revenue
            col2.subheader("Revenue")
            col2.success(str(supplier_info["Revenue"]) + " M€", icon="💰")

            # Non-compliance Rate
            col3.subheader("Non-Default")
            col3.warning(str(round(100 * supplier_info['Non-compliance Rate'])) + "%",icon="🧯")

            # Delivery on time
            col1.subheader("Delivery on Time")
            col1.info(str(supplier_info['Delivery Time'])+ " %",icon="🚚")

            # Risk Score
            col2.subheader("Risk Score")
            if supplier_info['Risk Score']>66:
                col2.success(supplier_info['Risk Score'],icon="🟢")
            elif supplier_info['Risk Score']>33:
                col2.success(supplier_info['Risk Score'],icon="🟡")
            else :
                col2.success(supplier_info['Risk Score'],icon="🟠")

            # Main Risk Factor
            col3.subheader("Risk Variation")
            if supplier_info['Risk Variation']>0:
                col3.warning(supplier_info['Risk Variation'],icon="🔺")
            else : 
                col3.warning(supplier_info['Risk Variation'],icon="🔻")

            # Delivery Time Trend over the last 12 months
            st.subheader("Delivery Trend " + supplier_name)
            # Create interactive chart with Plotly
            fig = px.line(df_delivery, x='Month', y='Delivery Time', title="Delivery On-Time Trend (D1) and Non-Default Delivery Trend per Month (N1)")
            fig.update_xaxes(title="Month")
            fig.update_yaxes(title="%")
            fig.add_trace(go.Scatter(x=df_delivery['Month'], y=df_delivery['Delivery Time'], mode='lines', name='D1'))
            fig.add_trace(go.Scatter(x=df_delivery['Month'], y=df_delivery['n1'], mode='lines', name='N1'))


            # Display interactive chart with Streamlit
            st.plotly_chart(fig)

    ## Onlget Graph
    elif tab_selected == "Graph":

        st.subheader("Graphe des fournisseurs")
        col1, col2 = st.columns([2, 1])

        ## pyvis
        
        #Network(notebook=True)
        #make Network show itself with repr_html
        df = got.simple_func(data)

        with col1:
            HtmlFile = open("test.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code,height=400)

        col2.dataframe(df)
        

        #def net_repr_html(self):
        #  nodes, edges, height, width, options = self.get_network_data()
        #  html = self.template.render(height=height, width=width, nodes=nodes, edges=edges, options=options)
        #  return html

        #Network._repr_html_ = net_repr_html
        # st.sidebar.title('Choose your favorite Graph')
        # #option=st.sidebar.selectbox('select graph',('Simple','Karate', 'GOT'))
        # physics=False #st.sidebar.checkbox('add physics interactivity?')
        # got.simple_func(physics)


        # HtmlFile = open("test.html", 'r', encoding='utf-8')
        # source_code = HtmlFile.read() 
        # components.html(source_code, height = 500,width=500)


        # # Affichage du graphe
        # plt.figure(figsize=(8, 6))
        # pos = nx.spring_layout(graph)
        # nx.draw_networkx(graph, pos=pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='gray')
        # col1.pyplot()

        # # Calcul du PageRank
        # pr = nx.pagerank(graph)
        # pr_df = pd.DataFrame.from_dict(pr, orient='index', columns=['PageRank'])
        # col2.dataframe(pr_df)

        # # Affichage du DataFrame des connexions
        # st.subheader("Connexions entre les fournisseurs")
        # st.dataframe(df_connections)


       