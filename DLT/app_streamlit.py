# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from PIL import Image

#import seaborn as sns
import plotly.express as px
from function_prediction import *
# Charger les données
données = pd.read_pickle("dataset_moteurs_enrichi.pkl")

# configuration de la page
st.set_page_config(layout="wide")

# Ajout du logo
img_logo = Image.open("Logo_Iod_solutions_Horizontal_Logo_Complet_Original_RVB_1186px@72ppi (1).png")
st.sidebar.image(img_logo)


page = st.sidebar.radio("Naviguer", ['Suivi de la performance industrielle', 'Suivi des anomalies','Prédiction'])
if page == 'Suivi de la performance industrielle':
    st.title('Suivi de la performance industrielle')
    # Sidebar pour les filtres
    st.sidebar.header('Filtres')

    # Ajoutez une option 'Tous' à vos sélections
    id_usine_options = ['All'] + list(données['Localisation_Usine'].unique())
    id_usine = st.sidebar.selectbox('Sélectionnez une Usine:', id_usine_options)

    type_moteur_options = ['All'] + list(données['Type_Moteur'].unique())
    type_moteur = st.sidebar.selectbox('Sélectionnez le type de moteur:', type_moteur_options)

    mode_expedition_options = ['All'] + list(données['Mode_Expédition'].unique())
    mode_expedition = st.sidebar.selectbox("Sélectionnez le mode d'expédition", mode_expedition_options)
    # Initialisez 'données_filtrées' avec 'données' pour inclure tous les cas par défaut
    données_filtrées = données

    # Appliquez les filtres seulement si l'option sélectionnée n'est pas 'Tous'
    if id_usine != 'All':
        données_filtrées = données_filtrées[données_filtrées['Localisation_Usine'] == id_usine]

    if type_moteur != 'All':
        données_filtrées = données_filtrées[données_filtrées['Type_Moteur'] == type_moteur]

    if mode_expedition != 'All':
        données_filtrées = données_filtrées[données_filtrées['Mode_Expédition'] == mode_expedition]

    # Analyse et visualisations des données...
    # Calcul des indicateurs clés
    nombre_total_productions = len(données_filtrées)
    temps_moyen_fabrication = données_filtrées['Durée_Fabrication'].mean()
    temps_median_fabrication = données_filtrées['Durée_Fabrication'].median()

    nombre_total_moteur = len(données_filtrées["Spécifications"].unique())
    temps_moyen_dlt = données_filtrées['DLT'].mean()
    temps_median_dlt = données_filtrées['DLT'].median()

    # Affichage des indicateurs clés dans des boxs avec des titres, valeurs et icônes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Nombre Total Productions 🏭", value=f"{nombre_total_productions}")
        
    with col2:
        st.metric(label="Temps Moyen Fabrication ⏳", value=f"{temps_moyen_fabrication:.2f} h")
        
    with col3:
        st.metric(label="Temps Médian Fabrication ⌛", value=f"{temps_median_fabrication:.2f} h")

    with col1:
        st.metric(label="Nombre Total Moteurs 🏭", value=f"{nombre_total_moteur}")
        
    with col2:
        st.metric(label="Temps Moyen DLT ⏳", value=f"{temps_moyen_dlt:.2f} h")
        
    with col3:
        st.metric(label="Temps Médian DLT ⌛", value=f"{temps_median_dlt:.2f} h")


    # Suivi de l'évolution du DLT
    st.header('Suivi de l\'évolution du DLT')
    # Assurez-vous que les dates sont au bon format
    données_filtrées['Date_Début_Fabrication'] = pd.to_datetime(données_filtrées['Date_Début_Fabrication'])
    données_filtrées.sort_values('Date_Début_Fabrication', inplace=True)

    données_filtrées['Date'] = pd.to_datetime(données_filtrées['Date_Début_Fabrication'])

    # Sélection du type de donnée à visualiser: DLT ou Temps de Fabrication
    type_donnée = st.radio("Sélectionnez le type de donnée à visualiser:", ('DLT', 'Temps de Fabrication'),horizontal=True)
    colonne_donnée = 'DLT' if type_donnée == 'DLT' else 'Durée_Fabrication'

    # Agrégation des données par mois et par localisation d'usine
    # Note : 'Grouper' permet de grouper par périodes; ici, nous utilisons 'M' pour mois
    données_mensuelles = données_filtrées.groupby([pd.Grouper(key='Date', freq='M'), 'Localisation_Usine'])[colonne_donnée].mean().reset_index()

    # Lissage des données agrégées si nécessaire (facultatif, dépend de la quantité de données et de la préférence visuelle)
    # Par exemple, utiliser une moyenne mobile simple sur les données agrégées
    données_mensuelles[f'{colonne_donnée}_Lissé'] = données_mensuelles.groupby('Localisation_Usine')[colonne_donnée].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Création du graphique interactif avec Plotly Express
    fig = px.line(données_mensuelles, x='Date', y=f'{colonne_donnée}_Lissé', color='Localisation_Usine',
                labels={f'{colonne_donnée}_Lissé': f'{type_donnée} (lissé, mensuel)'},
                title=f'Courbe lissée et agrégée du {type_donnée} par Localisation d\'Usine')

    # Améliorer la mise en forme du graphique
    fig.update_layout(xaxis_title='Date', yaxis_title=f'{type_donnée} (moyenne mensuelle)', xaxis=dict(rangeslider=dict(visible=True), type="date"))

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)



    # Classement des usines...
    # Affichage des classements
    st.subheader(f'Classement')

    col1, col2 = st.columns(2)
    # Option pour choisir le critère d'agrégation dans la sidebar
    with col1 : 
        option_agrégation = st.selectbox(
        'Choisissez la variable pour agréger les données:',
        ('Localisation_Usine', 'Type_Moteur', 'Spécifications'))

    # Choix des métriques pour le classement
    with col2 :
        option_metric = st.selectbox(
        'Choisissez la métrique pour le classement:',
        ('Temps de Fabrication', 'DLT'))

    # Filtrer les données en conséquence (en supposant que données_filtrées est déjà défini)
    # Calcul des moyennes pour le critère sélectionné
    if option_metric == 'Temps de Fabrication':
        moyennes = données_filtrées.groupby(option_agrégation)['Durée_Fabrication'].mean().sort_values(ascending=False)
    else:
        moyennes = données_filtrées.groupby(option_agrégation)['DLT'].mean().sort_values(ascending=False)



    # Définition de la colonne de données en fonction du choix de métrique
    colonne_donnée = 'Durée_Fabrication' if option_metric == 'Temps de Fabrication' else 'DLT'

    # Calcul des moyennes pour le critère sélectionné et tri pour le classement
    moyennes = données_filtrées.groupby(option_agrégation)[colonne_donnée].mean().sort_values(ascending=True).reset_index()

    # Création du graphique avec Plotly Express
    fig = px.bar(moyennes, y=option_agrégation, x=colonne_donnée, orientation='h',
                title=f'Classement par {option_agrégation} basé sur {option_metric}',
                labels={colonne_donnée: 'Moyenne', option_agrégation: option_agrégation},
                color=colonne_donnée,
                color_continuous_scale=px.colors.sequential.Viridis)

    # Amélioration de la mise en page du graphique
    fig.update_layout(xaxis_title='Moyenne', yaxis_title=option_agrégation)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

elif page=="Suivi des anomalies" :
    st.title('Suivi des anomalies')

    # Clustering...

    # Détection d'anomalies
    # Identification des anomalies selon les critères définis
    col1, col2 = st.columns(2)
    with col1 : 
        seuil_dlt = st.sidebar.number_input('Seuil pour le DLT (h)', min_value=0, value=40)
    with col2 : 
        seuil_fabrication = st.sidebar.number_input('Seuil pour la durée de fabrication (h)', min_value=0, value=25)
    données['Anomalie'] = (données['DLT'] > seuil_dlt) | (données['Durée_Fabrication'] > seuil_fabrication)
    
    # Conversion des dates en mois (format période) puis en chaîne de caractères pour la sérialisation
    données['Mois'] = données['Date_Début_Fabrication'].dt.to_period('M').astype(str)
    
    # Agrégation du nombre d'anomalies par mois et par usine
    anomalies_agrégées = données.groupby(['Mois', 'Localisation_Usine'])['Anomalie'].sum().reset_index()
    
    # Préparation des données pour la visualisation avec Plotly
    fig = px.bar(anomalies_agrégées, x='Mois', y='Anomalie', color='Localisation_Usine', 
                 labels={'Anomalie':'Nombre d\'anomalies'}, title='Nombre d\'anomalies par mois et par usine')
    fig.update_xaxes(categoryorder='category ascending')  # Assure l'ordre chronologique des mois
    col1, col2 = st.columns([2,1])
    with col1 :
        st.plotly_chart(fig,use_container_width=True)


    # Affichage du tableau récapitulatif
    anomalies_par_usine = données.groupby('Localisation_Usine')['Anomalie'].sum().reset_index()
    with col2 :
        st.dataframe(anomalies_par_usine)

    # Calcul du total de productions par mois et par usine pour le calcul des pourcentages
    total_productions = données.groupby(['Mois', 'Localisation_Usine']).size().reset_index(name='Total_Productions')

    # Fusion des anomalies agrégées avec le total de productions
    anomalies_pourcentage = pd.merge(anomalies_agrégées, total_productions, on=['Mois', 'Localisation_Usine'])
    
    # Calcul du pourcentage d'anomalies
    anomalies_pourcentage['Pourcentage_Anomalies'] = (anomalies_pourcentage['Anomalie'] / anomalies_pourcentage['Total_Productions']) * 100

    # Préparation des données pour la visualisation avec Plotly
    fig = px.line(anomalies_pourcentage, x='Mois', y='Pourcentage_Anomalies', color='Localisation_Usine',
                  labels={'Pourcentage_Anomalies':'Pourcentage d\'anomalies', 'Mois':'Mois'}, title='Pourcentage d\'anomalies par mois et par usine')
    fig.update_layout(xaxis_title='Mois', yaxis_title='Pourcentage d\'anomalies', xaxis=dict(rangeslider=dict(visible=True), type="date"))
    col1, col2 = st.columns([2,1])
    with col1 :
        st.plotly_chart(fig,use_container_width=True)
    anomalies_par_usine_pourc = données.groupby('Localisation_Usine')['Anomalie'].mean().reset_index()
    with col2 :
        st.dataframe(anomalies_par_usine_pourc)

else :
    # Prédictions...
    st.title('Prédiction et Détection d\'anomalies')
    # Calcul de la conformité pour chaque jour
    id_usine_options = ['All'] + list(données['Localisation_Usine'].unique())
    id_usine = st.sidebar.selectbox('Sélectionnez une Usine:', id_usine_options)
    
    if id_usine!='All':
        données_filtrées = données[données["Localisation_Usine"]==id_usine]
    else :
        données_filtrées = données.copy()

    last_month = données_filtrées['Date_Début_Fabrication'].dt.month.max()
    données_last_month = données_filtrées[données_filtrées['Date_Début_Fabrication'].dt.month == last_month]

    def check_compliance(row):
        return row['DLT_lower_bound'] <= row['DLT'] <= row['DLT_upper_bound']

    # Application de la vérification
    données_last_month['Compliance'] = données_last_month.apply(check_compliance, axis=1)

    # Calcul du pourcentage de conformité par jour
    compliance_daily = 100 - (données_last_month.groupby(données_last_month['Date_Début_Fabrication'].dt.date)['Compliance'].mean() * 100)
    compliance_daily = compliance_daily.sort_index()

    # Affichage des résultats
    st.write(f"Anomalie des DLT par jour pour le dernier mois ({last_month}):")
    col1,col2 = st.columns(2)
    for day, compliance in compliance_daily.items():
        if compliance < 85 :
            # Insère un émoticône de warning si la compliance est inférieure à 85%
            with col1 : 
                emoji = "⚠️"  # Émoticône de warning
                st.markdown(f"Date: `{day}` - Conformité: `{compliance:.2f}%` {emoji}", unsafe_allow_html=True)
        else:
            # Insère un "V" vert (coche) si la compliance est de 50% ou plus
            emoji = "✅"  # Émoticône de coche verte

    ## ajout d'un graphique 
    compliance_daily_df = pd.DataFrame({
        'Date': compliance_daily.index,
        'Conformité': compliance_daily.values
    })

    # Création du graphique de l'évolution de la conformité avec Plotly
    fig = px.line(compliance_daily_df, x='Date', y='Conformité',
                  title='Évolution de la Conformité sur le Dernier Mois',
                  labels={'Conformité': 'Pourcentage de Non-Conformité (%)', 'Date': 'Date'},
                  markers=True)  # Ajoute des marqueurs pour chaque point de donnée
    fig.add_hline(y=85, line_dash="dot",
                  annotation_text="Seuil de 85%",
                  annotation_position="bottom right",
                  line_color="red")

    # Amélioration de la mise en forme du graphique
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Pourcentage de Non-Conformité (%)',
                      xaxis=dict(rangeslider=dict(visible=True), type="date"))

    # Affichage du graphique dans Streamlit
    with col2 : 
        st.plotly_chart(fig, use_container_width=True)

    # Affichage 
    # Assurez-vous d'avoir importé matplotlib.pyplot comme plt

    # Calcul du pourcentage d'anomalies par usine
    pourcentage_anomalies_par_usine = données.groupby('Localisation_Usine')['Incidents_Fabrication'].mean() * 100

    # Calcul du DLT moyen par usine
    dlt_moyen_par_usine = données.groupby('Localisation_Usine')['DLT'].mean()

    # Création d'un DataFrame pour le graphique
    data_for_plot = pd.DataFrame({
        'Pourcentage d\'anomalies': pourcentage_anomalies_par_usine,
        'DLT moyen': dlt_moyen_par_usine
    }).reset_index()

    # Création du graphique en utilisant Plotly Express
    fig = px.scatter(data_for_plot, 
                    x='Pourcentage d\'anomalies', 
                    y='DLT moyen', 
                    text='Localisation_Usine',
                    size='DLT moyen', # Optionnel: ajuste la taille des points en fonction du DLT moyen
                    hover_data=['Localisation_Usine'], # Affiche le nom de l'usine lorsque vous passez la souris sur un point
                    title='Pourcentage d\'anomalies vs DLT moyen par usine')

    fig.update_traces(textposition='top center') # Ajuste la position du texte pour la lisibilité
    fig.update_layout(xaxis_title='Pourcentage d\'anomalies (%)',
                    yaxis_title='DLT moyen (heures)',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True))

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


        

            

