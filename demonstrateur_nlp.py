# app.py

import streamlit as st
from joblib import load
# Votre code des fonctions ...
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

nlp = spacy.load('fr_core_news_sm')



def display_results(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Train Precision:", precision_score(y_train, y_train_pred))
    print("Train Recall:", recall_score(y_train, y_train_pred))
    print("Test Precision:", precision_score(y_test, y_test_pred))
    print("Test Recall:", recall_score(y_test, y_test_pred))
    
    # ROC and AUC
    train_probs = model.predict_proba(X_train)[:,1]
    test_probs = model.predict_proba(X_test)[:,1]
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    print('Train AUC:', train_auc)
    print('Test AUC:', test_auc)
    
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_fpr, train_tpr, label='Train AUC: {:.2f}'.format(train_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Train ROC Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_fpr, test_tpr, label='Test AUC: {:.2f}'.format(test_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized_text


# # Liste des mots en rapport à la prise de rdv
# list_mots_rdv = ["rendez-vous","rencontre","entrevue","consultation","entretien","réunion","rdv","appo","Rdv","disponibilité","créneau","horaire","date","heure"]
# def predict_proba_for_text(text, model, vectorizer, nlp):
#     # Prétraitement du texte
#     preprocessed_text = preprocess_text(text)

#     # Transformation TF-IDF
#     tfidf_features = vectorizer.transform([preprocessed_text])

#     # Prédiction avec le modèle
#     proba = model.predict_proba(tfidf_features)[0][1]  # prend la probabilité pour la classe 1
#     return proba

list_mots_rdv = ["rendez-vous","rencontre","entrevue","consultation","entretien","réunion","rdv","appo","Rdv","disponibilité","créneau","horaire","date","heure"]

def adjust_proba(proba, factor=0.4, increase=True):
    if increase:
        return proba + (1 - proba) * factor
    else:
        return proba * (1 - factor)

def contains_keyword(text, keywords):
    for word in keywords:
        if word in text:
            return True
    return False

def predict_proba_for_text(text, model, vectorizer, nlp):
    # Prétraitement du texte
    preprocessed_text = preprocess_text(text)

    # Transformation TF-IDF
    tfidf_features = vectorizer.transform([preprocessed_text])

    # Prédiction avec le modèle
    proba = model.predict_proba(tfidf_features)[0][1]  # prend la probabilité pour la classe 1

    # Ajustement de la probabilité basée sur la présence/absence de mots-clés
    if contains_keyword(preprocessed_text, list_mots_rdv):
        proba = adjust_proba(proba, increase=True)
    else:
        proba = adjust_proba(proba, increase=False)
    
    return proba


# Chargement du modèle sauvegardé et du vectorizer
loaded_model = load('NLP/best_rf_model.joblib')
vectorizer = load('NLP/tfidf_vectorizer.joblib')  # Supposez que vous avez également sauvegardé votre vectorizer
nlp = spacy.load('fr_core_news_sm')

st.title("Prédicteur de rendez-vous")

# Entrée utilisateur
user_input = st.text_area("Entrez votre texte ici:")

if st.button('Prédire'):
    # Prédiction de la probabilité à l'aide de la fonction fournie
    proba = predict_proba_for_text(user_input, loaded_model, vectorizer, nlp)
    
    # Affichage de la probabilité que le texte soit un rendez-vous
    st.write(f"Probabilité que le texte soit un rendez-vous: {proba:.2f}")

# Pour exécuter l'application, utilisez: streamlit run app.py
