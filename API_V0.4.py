import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.pyfunc
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
import shap
from lightgbm import LGBMClassifier
import plotly.graph_objects as go

model_path = "mlflow_model" 
file_path = "Input_data/"

# Chargement des données / modèle / masque
@st.cache_resource
def load_all():
    model = mlflow.lightgbm.load_model(model_path)
    df = pd.read_csv(file_path + "Dataframe_20.csv")    
    return model, df

# Charger le modèle et le masque des colonnes
model, df = load_all()

def plot_gauge(probability, cutoff):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100, 
        title={'text': "Probabilité de Défaut (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "grey"},
            'steps': [
                {'range': [0, cutoff * 100], 'color': "lightgreen"},
                {'range': [cutoff * 100, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': cutoff * 100
            }
        }
    ))
    return fig

# Interface utilisateur
st.title("Application de prédiction du défaut client")
st.write("Cette application utilise un modèle MLflow pour faire des prédictions à partir d'un identifiant client.")

# Demander l'identifiant du client
sk_id_curr = st.text_input("Entrez l'identifiant client (SK_ID_CURR) :", "")

if sk_id_curr:
    try:
        sk_id_curr = int(sk_id_curr)
    except ValueError:
        sk_id_curr = -1
        
    client_data = df[df["SK_ID_CURR"] == sk_id_curr].drop(columns=['TARGET'])

    if client_data.empty:
        st.error("Aucun client trouvé avec cet identifiant.")
    else:
        st.write("Données du client sélectionné :", client_data)

    try:
        prediction = model.predict(client_data)
        probability = model.predict_proba(client_data)[:, 1]

        # Affichage de la jauge de probabilité
        cutoff = 0.07
        st.plotly_chart(plot_gauge(probability[0], cutoff))
        st.write("Probabilité de défaut :", probability[0] * 100, "%")
        st.write("La demande de prêt du client doit être :")
        if probability > cutoff:
            st.markdown("<h2 style='color:red;'>Refusé</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Accepté</h2>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

    try:
        st.write("Examen de la feature importance du client")
        # Initialiser SHAP explainer pour le modèle LightGBM
        explainer = shap.TreeExplainer(model)

        # Calculer les valeurs SHAP pour le client unique dans client_data
        shap_values = explainer.shap_values(client_data)

        # Vérifier si shap_values est une liste (sortie binaire) et sélectionner la première classe
        shap_values_single = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Base value correspondante
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

        # Afficher le graphique en cascade pour le client unique
        plt.figure()
        shap.waterfall_plot(shap.Explanation(values=shap_values_single[0], 
                                             base_values=base_value, 
                                             data=client_data.iloc[0]))
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"Erreur lors de la feature importance : {e}")
else:
    st.write("Veuillez entrer un identifiant client pour commencer les prédictions.")
