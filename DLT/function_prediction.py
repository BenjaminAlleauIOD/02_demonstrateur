import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer personnalisé pour convertir une colonne datetime en caractéristiques numériques."""
    def __init__(self, date_col):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['Year'] = X_[self.date_col].dt.year
        X_['Month'] = X_[self.date_col].dt.month
        X_['Day'] = X_[self.date_col].dt.day
        return X_.drop(columns=[self.date_col])

def detecter_anomalies(df, date_col, target, cat_attribs):
    """
    Détecte les anomalies dans les prédictions de la variable cible pour le dernier mois
    en utilisant les variables catégorielles et de date.

    Paramètres :
    - df (pd.DataFrame) : DataFrame contenant les données.
    - date_col (str) : Nom de la colonne contenant les dates.
    - target (str) : Nom de la colonne de la variable cible à prédire.
    - cat_attribs (list) : Liste des noms des colonnes catégorielles.

    Retourne :
    - pd.DataFrame contenant les données des anomalies détectées.
    """
    
    # Filtrer les données pour le dernier mois
    dernier_mois = df[date_col].max().month
    dernières_données = df[df[date_col].dt.month == dernier_mois]

    # Préparer les features en excluant la cible
    features = dernières_données.drop(columns=[target])
    y = dernières_données[target]

    # Création du préprocesseur avec OneHotEncoder et DateFeatureExtractor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(), cat_attribs)
    ], remainder='passthrough')

    # Construction du pipeline avec prétraitement et régression linéaire
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    pipeline.fit(X_train, y_train)

    # Prédiction des valeurs sur l'ensemble de test
    y_pred = pipeline.predict(X_test)

    # Calcul de l'erreur de prédiction
    erreur = y_test - y_pred

    # Utilisation d'Isolation Forest pour identifier les anomalies
    iso_forest = IsolationForest(contamination=0.05)
    anomalies = iso_forest.fit_predict(erreur.reshape(-1, 1))
    indices_anomalies = np.where(anomalies == -1)[0]

    # Récupération et retour des données correspondantes aux anomalies
    données_anomalies = X_test.iloc[indices_anomalies]
    return données_anomalies.reset_index()

# Note : Adaptez 'cat_attribs' avec les vrais noms de vos colonnes catégorielles
# anomalies_détectées = detecter_anomalies(données, 'Date_Début_Fabrication', 'DLT', ['Type_Moteur', 'Spécifications', 'Localisation_Usine', 'Mode_Expédition'])
# print(anomalies_détectées)
