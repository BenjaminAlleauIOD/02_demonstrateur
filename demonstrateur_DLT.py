import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

# lecture data
df = pd.read_pickle("DLT/data_fictive_dlt.pkl")
df['Date'] = pd.to_datetime(df['Date'])

# Titre
st.title('Analyse des DLT')

# Menu déroulant pour le choix de l'onglet
option = st.sidebar.selectbox('Choisir un onglet', ['Global', 'Filtre','Comparaison par dimension'])

# Si l'utilisateur choisit "Global"
if option == 'Global':
    # KPIs
    st.subheader("Indicateurs clés de performance (KPI)")

    # Calculer les KPIs
    avg_dlt = df['DLT'].mean()
    max_dlt = df['DLT'].max()
    min_dlt = df['DLT'].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("DLT Moyen", f"{avg_dlt:.2f} jours")
    col2.metric("DLT le plus long", f"{max_dlt} jours")
    col3.metric("DLT le plus court", f"{min_dlt} jours")

    st.subheader("Temps moyen par étape en jour(s)")
    col1, col2, col3,col4, col5, col6,col7 = st.columns(7)
    col1.metric("Attente", "1")
    col2.metric("Planif", f"10")
    col3.metric("Validation", f"4")
    col4.metric("Lancement", f"3")
    col5.metric("Fabrication", f"4")
    col6.metric("Vérification", f"3")
    col7.metric("Libération", f"1")

    st.subheader("Courbe de densité globale des DLTs")

    # Création de la figure
    fig = go.Figure()

    # Utiliser toutes les données DLT pour estimer la densité
    all_dlt_data = df['DLT']
    density = gaussian_kde(all_dlt_data)
    x_vals = np.linspace(min(all_dlt_data), max(all_dlt_data), 1000)
    y_vals = density(x_vals)

    # Ajouter la trace à la figure
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="DLT Global"))

    # Mise à jour des paramètres de la figure pour améliorer le rendu
    fig.update_layout(barmode='overlay', bargap=0.1)
    fig.update_traces(opacity=0.6)

    # Affichage de la figure dans Streamlit
    st.plotly_chart(fig)


    

   
    # Histogramme
    st.subheader("Histogramme des DLTs")
    fig = go.Figure(data=[go.Histogram(x=df['DLT'])])
    st.plotly_chart(fig)

    # Pie Chart
    st.subheader("Répartition des DLTs")

    dlt_ranges = ["10-20 jours", "20-30 jours", "30+ jours"]
    dlt_counts = [
        df[(df['DLT'] >= 10) & (df['DLT'] <= 20)].shape[0],
        df[(df['DLT'] > 20) & (df['DLT'] <= 30)].shape[0],
        df[df['DLT'] > 30].shape[0]
    ]

    fig = go.Figure(data=[go.Pie(labels=dlt_ranges, values=dlt_counts)])
    fig.update_layout(margin=dict(t=50, b=50, r=50, l=50))
    st.plotly_chart(fig)

    # Line chart for DLT Evolution by month
    df['Month'] = df['Date'].dt.month
    monthly_avg_dlt = df.groupby('Month')['DLT'].mean()
    st.subheader("Évolution du DLT par mois pour l'année 2023")
    fig = go.Figure(data=go.Scatter(x=monthly_avg_dlt.index, y=monthly_avg_dlt.values, mode='lines+markers'))
    st.plotly_chart(fig)


# Si l'utilisateur choisit "Filtre"
elif option == 'Filtre':
    st.subheader("Filtrage des DLTs")
    col1, col2 = st.columns(2)
    # Dropdown pour le filtre CAD
    unique_cad = df['CAD'].unique()
    selected_cad = col1.selectbox('Sélectionnez un CAD', ['Tous'] + list(unique_cad))
    if selected_cad != 'Tous':
        df = df[df['CAD'] == selected_cad]

    # Dropdown pour le filtre Dimension
    unique_dimension = df['Dimension'].unique()
    selected_dimension = col2.selectbox('Sélectionnez une Dimension', ['Tous'] + list(unique_dimension))
    if selected_dimension != 'Tous':
        df = df[df['Dimension'] == selected_dimension]

    # Reproduire les graphiques similaires à l'onglet 'Global'
    # KPIs
    st.subheader("Indicateurs clés de performance (KPI)")

    # Calculer les KPIs
    avg_dlt = df['DLT'].mean()
    max_dlt = df['DLT'].max()
    min_dlt = df['DLT'].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("DLT Moyen", f"{avg_dlt:.2f} jours")
    col2.metric("DLT le plus long", f"{max_dlt} jours")
    col3.metric("DLT le plus court", f"{min_dlt} jours")

    st.subheader("Temps moyen par étape en jour(s)")
    col1, col2, col3,col4, col5, col6,col7 = st.columns(7)
    col1.metric("Attente", "1")
    col2.metric("Planif", f"10")
    col3.metric("Validation", f"4")
    col4.metric("Lancement", f"3")
    col5.metric("Fabrication", f"4")
    col6.metric("Vérification", f"3")
    col7.metric("Libération", f"1")

    # Pie Chart
    st.subheader("Répartition des DLTs")

    dlt_ranges = ["10-20 jours", "20-30 jours", "30+ jours"]
    dlt_counts = [
        df[(df['DLT'] >= 10) & (df['DLT'] <= 20)].shape[0],
        df[(df['DLT'] > 20) & (df['DLT'] <= 30)].shape[0],
        df[df['DLT'] > 30].shape[0]
    ]

    fig = go.Figure(data=[go.Pie(labels=dlt_ranges, values=dlt_counts)])
    st.plotly_chart(fig)

    # Histogramme
    st.subheader("Histogramme des DLTs")
    fig = go.Figure(data=[go.Histogram(x=df['DLT'], histnorm='probability density')])
    st.plotly_chart(fig)

  
    df['Month'] = df['Date'].dt.month
    monthly_avg_dlt = df.groupby('Month')['DLT'].mean()


    # Line chart for DLT Evolution by month
    st.subheader("Évolution du DLT par mois pour l'année 2023")
    fig = go.Figure(data=go.Scatter(x=monthly_avg_dlt.index, y=monthly_avg_dlt.values, mode='lines+markers'))
    st.plotly_chart(fig)

elif option == 'Comparaison par dimension':
    st.subheader("Comparaison des DLTs par Dimension")

    # Box Plot par dimension
    fig = go.Figure()

    for dimension in df['Dimension'].unique():
        fig.add_trace(go.Box(y=df[df['Dimension'] == dimension]['DLT'], name=str(dimension)))

    fig.update_layout(title="Distribution des DLT par dimension", xaxis_title="Dimension", yaxis_title="DLT", margin=dict(t=50, b=50, r=50, l=50))
    st.plotly_chart(fig)

    unique_dimensions = df['Dimension'].unique()
    
    # DLT Moyen par dimension
    avg_dlt_by_dim = df.groupby('Dimension')['DLT'].mean()
    st.subheader("DLT Moyen par Dimension")
    fig = go.Figure(data=go.Bar(x=avg_dlt_by_dim.index, y=avg_dlt_by_dim.values))
    st.plotly_chart(fig)

    # Évolution du DLT par mois pour chaque dimension
    df['Month'] = df['Date'].dt.month
    monthly_avg_dlt_by_dim = df.groupby(['Dimension', 'Month'])['DLT'].mean().unstack()
    st.subheader("Évolution du DLT par mois par Dimension pour l'année 2023")
    fig = go.Figure()
    for dim in unique_dimensions:
        fig.add_trace(go.Scatter(x=monthly_avg_dlt_by_dim.columns, y=monthly_avg_dlt_by_dim.loc[dim], mode='lines+markers', name=str(dim)))
        #fig.add_trace(go.Scatter(x=monthly_avg_dlt_by_dim.columns, y=monthly_avg_dlt_by_dim.loc[dim], mode='lines+markers', name=dim))
    st.plotly_chart(fig)

    # Courbe de densité (KDE) pour chaque dimension
    st.subheader("Courbes de densité des DLTs par Dimension")
 
    # Création de la figure
    fig = go.Figure()

    # Pour chaque dimension unique
    unique_dimensions = df['Dimension'].unique()
    for dim in unique_dimensions:
        # Filtrer les données pour cette dimension
        dim_data = df[df['Dimension'] == dim]['DLT']
        
        # Estimation de la densité
        density = gaussian_kde(dim_data)
        x_vals = np.linspace(min(dim_data), max(dim_data), 1000)
        y_vals = density(x_vals)
        
        # Ajouter la trace à la figure
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=str(dim)))

    # Mise à jour des paramètres de la figure pour améliorer le rendu
    fig.update_layout(barmode='overlay', bargap=0.1)
    fig.update_traces(opacity=0.6)

    # Affichage de la figure dans Streamlit
    st.plotly_chart(fig)


    # Nombre de CAD par dimension
    cad_count_by_dim = df.groupby('Dimension')['CAD'].nunique()
    st.subheader("Nombre de CAD par Dimension")
    fig = go.Figure(data=go.Bar(x=cad_count_by_dim.index, y=cad_count_by_dim.values))
    st.plotly_chart(fig)

