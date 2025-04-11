import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import streamviz
import pandas as pd
import plotly.express as px
import joblib
import shap
import plotly.graph_objects as go




# URL de l'API deployée sur heroku
API_URL = "https://application-prediction-scoring-b81541cc2c3b.herokuapp.com/predict"

probability = None
# charger les données
df_sample = pd.read_csv('df_sample.csv')
# définir SK_ID_CUR en index
df_sample = df_sample.set_index('SK_ID_CURR')
# charger le scaler
scaler = joblib.load('scaler.pkl')
# charger le modele
modele_retenu = joblib.load('modele_retenu.pkl')
# charger l'explaineur shap
shap_explainer = joblib.load('global_explainer.pkl')

df_pour_streamlit = pd.read_csv('df_pour_streamlit.csv')
df_pour_streamlit = df_pour_streamlit.set_index('SK_ID_CURR')
df_pour_streamlit.loc[df_pour_streamlit["DAYS_BIRTH"] == 365243, "DAY_BIRTH"] = np.nan
df_pour_streamlit.loc[df_pour_streamlit["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
df_pour_streamlit["AGE"] = np.floor(df_pour_streamlit["DAYS_BIRTH"] / -365).fillna(0).astype(int)
df_pour_streamlit["YEARS_EMPLOYED"] = np.floor(df_pour_streamlit["DAYS_EMPLOYED"] / -365).fillna(0).astype(int)

liste_variable_principales = ["CODE_GENDER",
                              "FLAG_OWN_CAR",
                              "FLAG_OWN_REALTY",
                              "CNT_CHILDREN",
                              "AMT_INCOME_TOTAL",
                              "AMT_CREDIT",
                              "AMT_ANNUITY",
                              "AMT_GOODS_PRICE",
                              "AGE",
                              "YEARS_EMPLOYED"]
df_streamlit_filtered = df_pour_streamlit[liste_variable_principales]
transposed_df_streamlit_filtered = df_streamlit_filtered.T

# récupérer le meilleur seuil retenu
meilleur_seuil_value = 0.5050505050505051

X = df_sample.drop(columns=['TARGET'])

X_scaled = scaler.transform(X)
# definir un df avec X_scaled
X_scaled_df = pd.DataFrame(X_scaled,
                           index=X.index,
                           columns=X.columns)

st.title("Prédiction du score de crédit pour le client")

st.sidebar.image("16794938722698_Data Scientist-P7-01-banner.png")

client_id = st.session_state.get("client_id", None)

# Vérification que client_id existe, n'est pas vide et est un nombre entier
if not client_id or not client_id.isdigit():
    st.warning("Veuillez d'abord sélectionner un client sur la page d'accueil.")
    if st.button("Retour à l'accueil ..."):
        st.switch_page("Accueil.py")
    st.stop()





#client_transposed_df_streamlit_filtered = transposed_df_streamlit_filtered.loc(int(client_id))
df_info_client = df_streamlit_filtered.loc[[int(client_id)]]
transposed_df_info_client = df_info_client.T

col1, col2 = st.columns(2)
with col1:
        # Fonction pour récupérer la prédiction de l'API FastAPI
        def get_prediction(client_id):
            try:
                # Faire une requête GET à l'API
                response = requests.get(f"{API_URL}/{client_id}")
                if response.status_code == 200:
                    data = response.json()
                    return data
                elif response.status_code == 404:
                    st.error("Client non trouvé, renseignez un numéro client")
                    return None
                else:
                    st.error("Erreur dans la requête à l'API")
                    return None
            except Exception as e:
                st.error(f"Erreur : {e}")
                return None
        if client_id:
            # vérifier si le client_id est valide
            if not client_id.isdigit():
                st.error("Le numéro client doit être un nombre entier, renseignez un numéro de client")
            elif len(client_id) != 6:
                st.error("Le numéro client doit avoir exactement 6 chiffres, renseignez un numéro de client")
            else:
                # convertir en entier
                client_id_int = int(client_id)    
                # Appeler la fonction pour obtenir la prédiction
                result = get_prediction(client_id_int)
            
                if result is not None:
                    prediction_classe = result["prediction"]
                    probability = result["probability"]
                    st.write(f"**Prédiction :** {'Refusé' if prediction_classe == 1 else 'Accepté'}")
                    st.write(f"**Probabilité de refus :** {probability:.2%}")  
                    st.divider()
                    st.dataframe(transposed_df_info_client)
        else:
            st.write("Veuillez entrer un numéro de client.")

with col2:
    if probability is not None:
        # création de la figure
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Probabilité d'incident de paiement du client"},
            domain={'x': [0, 1], 'y': [0, 1]}, 
            gauge={
                'axis': {
                    'range': [0.0, 1.0],
                    'showticklabels': True,
                    'ticks': "",
                    'tickwidth': 0,
                    'tickcolor': "white"
                },
                'bgcolor': "white",
                'bar': {'color': "grey"},
                'steps': [
                    {'range': [0.0, (meilleur_seuil_value - 0.1)], 'color': "palegreen"},
                    {'range': [(meilleur_seuil_value - 0.1), (meilleur_seuil_value + 0.1)], 'color': "navajowhite"},
                    {'range': [(meilleur_seuil_value + 0.1), 1.0], 'color': "darksalmon"}
                ],
                'threshold': {
                    'line': {'color': "orangered", 'width': 4},
                    'thickness': 1,
                    'value': meilleur_seuil_value
                }
            }
        ))

        # layout propre sans axes parasites
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=50, b=0, l=0, r=0),
        )

        # Affichage dans Streamlit
        st.plotly_chart(fig)

        # Légende personnalisée HTML (propre, alignée, lisible)
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 10px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 15px; height: 15px; background-color: palegreen; margin-right: 5px;"></div>
                Zone de faible risque
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 15px; height: 15px; background-color: navajowhite; margin-right: 5px;"></div>
                Zone intermédiaire
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 15px; height: 15px; background-color: darksalmon; margin-right: 5px;"></div>
                Zone de risque élevé
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 15px; height: 4px; background-color: orangered; margin-right: 5px;"></div>
                Seuil de décision
            </div>
        </div>
        """, unsafe_allow_html=True)

if st.button("retour à l'accueil"):
    st.switch_page("Accueil.py")
if st.button("aller à l'explicabilité"):
    st.switch_page("pages/2_Explicabilité.py")