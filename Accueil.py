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

# récupérer le meilleur seuil retenu
meilleur_seuil_value = 0.5050505050505051

X = df_sample.drop(columns=['TARGET'])

X_scaled = scaler.transform(X)
# definir un df avec X_scaled
X_scaled_df = pd.DataFrame(X_scaled,
                           index=X.index,
                           columns=X.columns)

# calculer les valeurs shap pour toutes les observations
shap_values = shap_explainer(X_scaled_df)

# Configuration de la page principale
st.set_page_config(page_title="Dashboard scoring Crédit", layout="wide")

# Titre de l'application
st.title("Dashboard de Scoring de Crédit")

# Message d'introduction
st.write("""
Bienvenue sur l'application de scoring de crédit de la société Prêt à Dépenser  \n
         """)
st.caption("Renseignez un numéro client ci dessous, puis utilisez le menu de gauche pour naviguer entre les différentes sections :")
if "client_id" not in st.session_state:
    st.session_state.client_id = None

# Saisie du numéro de client
selected_client = st.text_input(label="Numéro client",
                                help="le numéro client est composé de 6 chiffres",
                                placeholder="Entrez le numéro client ...",
                                max_chars=6,
                                label_visibility="collapsed")
st.session_state.client_id = selected_client
if selected_client:
    if selected_client.isdigit() and len(selected_client) == 6:
        client_id_int = int(selected_client)
        if client_id_int in df_sample.index:
            st.session_state.client_id = selected_client
            st.success(f"Client sélectionné : {selected_client}")
        else:
            st.error("Ce numéro client n'existe pas dans les données.")
    else:
        st.error("Le numéro client doit contenir exactement 6 chiffres.")
#st.write(f"Client sélectionné : {st.session_state.client_id}")




st.page_link("pages/1_Prédiction.py", label="**Prédiction** : Obtenez une prédiction pour un client donné.")
st.page_link("pages/2_Explicabilité.py", label="**Explicabilité** : Comprenez pourquoi un client a une certaine prédiction.")
st.page_link("pages/3_Analyse.py", label="**Analyse** : Explorez les données et identifiez des tendances.")

# Afficher un client au hasard
client_au_hasard = df_sample.sample(1, random_state=None).index[0]
st.write(f"Exemple de client : {client_au_hasard}")

with st.expander("ou voir quelques exemples de numéros client disponibles"):
    st.write("Voici quelques exemples de clients présents dans la base :")
    exemples_clients = df_sample.sample(5, random_state=42).index.tolist()
    for client in exemples_clients:
        st.write(client)





# Lien vers la sidebar (géré automatiquement par Streamlit avec le dossier `pages/`)
st.sidebar.success("Renseignez un numéro client sur la page d'accueil puis sélectionnez une page dans le menu.")
st.sidebar.image("16794938722698_Data Scientist-P7-01-banner.png")



