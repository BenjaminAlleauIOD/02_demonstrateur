import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


## Fonction
def generate_fake_data():
    num_suppliers = 100
    suppliers = ['F ' + str(i) for i in range(1, num_suppliers+1)]
    addresses = ['Adresse ' + str(i) for i in range(1, num_suppliers+1)]
    delivery_times = np.random.randint(1, 10, num_suppliers)
    risk_scores = np.random.randint(1, 100, num_suppliers)
    ca = np.random.randint(1, 100, num_suppliers)
    compliance_rate_ind = np.random.uniform(0.5, 1.0, num_suppliers)
    risk_factors = np.random.choice(['Prix élevé', 'Retards fréquents', 'Mauvaise qualité'], num_suppliers)
    compliance_rate = np.random.uniform(0.5, 1.0)
    nb_retard = np.random.randint(0, 4, num_suppliers)
    
    current_date = datetime.now()
    date_range = pd.date_range(end=current_date, periods=num_suppliers)
    start_date = date_range.min()
    end_date = date_range.max()
    
    data = pd.DataFrame({'Fournisseur': suppliers,
                         'Adresse': addresses,
                         'Temps de livraison': delivery_times,
                         'Score de risque': risk_scores,
                         "Chiffre d'affaires": ca,
                         'Facteur de risque': risk_factors,
                         'Taux de non-conformité': compliance_rate_ind,
                         'Nombre de retard':nb_retard,
                         'Date': date_range})
    
    # Génération des données retard
    months = pd.date_range(start='2022-01-01', periods=12, freq='M')
    retard_counts = np.random.randint(low=20, high=100, size=12)

    # Création du DataFrame
    df_retard = pd.DataFrame({'Month': months, 'Retard Count': retard_counts})

    # Génération des données retard
    months = pd.date_range(start='2022-01-01', periods=12, freq='M')
    delai = np.random.randint(low=20, high=100, size=12)

    # Création du DataFrame
    df_livraison = pd.DataFrame({'Month': months, 'delai': delai})
    
    return data, compliance_rate, start_date, end_date,df_retard,df_livraison



## chargement des données factices 
data, compliance_rate, start_date, end_date,df_retard,df_livraison = generate_fake_data()

# Exécution de l'application
if __name__ == '__main__':

    # Chargement du logo depuis un fichier image
    logo_image = "image\logo-IoD-solutions.webp"

    # Affichage du logo dans la barre latérale
    st.sidebar.image(logo_image, width=200)


    st.sidebar.title("")
    tab_selected = st.sidebar.radio("Navigation", options=["Vue d'ensemble", "Fournisseurs"])

    # Filtrer les données en fonction de la période sélectionnée
    selected_data = data  # Afficher toutes les données

    # Configuration de l'application Streamlit
    st.title("Tableau de bord des fournisseurs 2023")

    if tab_selected == "Vue d'ensemble":
        
        # Affichage des informations générales
        st.subheader("Informations générales")

        # Nombre de fournisseurs
        col1, col2, col3 = st.columns(3)
        num_suppliers = len(selected_data)
        col1.info(f"Nombre de fournisseurs: {num_suppliers}")

        # Temps moyen de livraison
        avg_delivery_time = selected_data['Temps de livraison'].mean()
        col2.success(f"Temps moyen de livraison (en jours): {avg_delivery_time}")

        # Nombre de livraisons en retard
        sum_retard = selected_data['Nombre de retard'].sum()
        col3.warning(f"Nombre de livraisons en retard: {sum_retard}")

        # Les 5 fournisseurs les plus à risque
        col1, col2 = st.columns(2)
        col1.subheader("Les 5 fournisseurs les plus à risque")
        top_risk_suppliers = selected_data.nlargest(5, 'Score de risque')
        col1.dataframe(top_risk_suppliers[["Fournisseur","Score de risque","Facteur de risque"]],hide_index =True)

        # Taux de conformité
        with col2:
            jauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = round(100*compliance_rate),
        mode = "gauge+number+delta",
        title = {'text': "Taux de conformité"},
        delta = {'reference': 80},
        gauge = {'axis': {'range': [None, 100]},
                'steps' : [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 65}}))
            jauge = jauge.update_layout(autosize=True)


            st.plotly_chart(jauge,theme="streamlit", use_container_width=True)

        # graphique de l'évolution des retards 
        
        #st.subheader("Evolution du nombre de retards")
        # Création du graphique interactif avec Plotly
        fig = px.line(df_retard, x='Month', y='Retard Count', title="Evolution du nombre de retard")
        fig.update_xaxes(title="Mois")
        fig.update_yaxes(title="Nombre de retards")

        # Affichage du graphique interactif avec Streamlit
        st.plotly_chart(fig)

        
        

    elif tab_selected == "Fournisseurs":
        st.subheader("Fournisseurs")

        # Sélection d'un fournisseur
        supplier_name = st.selectbox("Sélectionnez un fournisseur", selected_data['Fournisseur'].unique())

        if supplier_name:
            supplier_info = selected_data[selected_data['Fournisseur'] == supplier_name].iloc[0]
            col1, col2, col3 = st.columns(3)
            # Adresse
            col1.subheader("Adresse")
            col1.info(supplier_info['Adresse'])

            # Chiffre d'affaires
            col2.subheader("Chiffre d'affaires")
            col2.success(supplier_info["Chiffre d'affaires"],icon="💰")

            # Taux de non-conformité
            col3.subheader("% de conformité")
            col3.warning(round(100*supplier_info['Taux de non-conformité']))

            # Delivery on time
            col1.subheader("Delivery on time")
            col1.info(supplier_info['Temps de livraison'])

            # Score de risque
            col2.subheader("Score de risque")
            col2.success(supplier_info['Score de risque'])

            # Facteur de risque principal
            col3.subheader("Risque principal")
            col3.warning(supplier_info['Facteur de risque'])

            # Évolution du temps de délai sur les 12 derniers mois
            st.subheader("Evolution du temps de livraison")
            # Création du graphique interactif avec Plotly
            fig = px.line(df_livraison, x='Month', y='delai', title="Évolution du temps de livraison par mois")
            fig.update_xaxes(title="Mois")
            fig.update_yaxes(title="Délai de livraison (jours)")

            # Affichage du graphique interactif avec Streamlit
            st.plotly_chart(fig)
