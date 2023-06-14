import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_fake_data():
    num_suppliers = 100
    suppliers = ['Fournisseur ' + str(i) for i in range(1, num_suppliers+1)]
    delivery_times = np.random.randint(1, 10, num_suppliers)
    risk_scores = np.random.randint(1, 100, num_suppliers)
    risk_factors = np.random.choice(['Prix élevé', 'Retards fréquents', 'Mauvaise qualité'], num_suppliers)
    compliance_rate = np.random.uniform(0.5, 1.0)
    
    current_date = datetime.now()
    date_range = pd.date_range(end=current_date, periods=num_suppliers)
    start_date = date_range.min()
    end_date = date_range.max()
    
    data = pd.DataFrame({'Fournisseur': suppliers,
                         'Temps de livraison': delivery_times,
                         'Score de risque': risk_scores,
                         'Facteur de risque': risk_factors,
                         'Date': date_range})
    
    return data, compliance_rate, start_date, end_date