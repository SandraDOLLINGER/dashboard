import streamlit as st
import pandas as pd

# charger les données
df_sample = pd.read_csv('df_sample.csv')
# définir SK_ID_CUR en index
df_sample = df_sample.set_index('SK_ID_CURR')

# Configuration de la page principale
st.set_page_config(page_title="Dashboard scoring Crédit", layout="wide")

# Titre de l'application
st.title("Dashboard de Scoring de Crédit",
         help="**À propos de ce tableau de bord** : Cette interface dynamique vous permet d’évaluer le risque de crédit d’un client grâce à un outil de prédiction automatisé. En renseignant un numéro client, vous accédez à une estimation personnalisée, des éléments d’explication, et des comparaisons avec d'autres profils similaires.")

# Message d'introduction
st.write("Bienvenue sur l'application de scoring de crédit de la société Prêt à Dépenser")
st.caption("Renseignez un numéro client ci dessous, puis utilisez le menu de gauche pour naviguer entre les différentes sections :")

# initialisation du client
if "client_id" not in st.session_state:
    st.session_state.client_id = None

# Vérification du changement de client et réinitialisation de la prédiction si nécessaire
if "prediction" in st.session_state:
    previous_client_id = st.session_state.get("client_id", None)
else:
    previous_client_id = None

# Saisie du numéro de client
selected_client = st.text_input(label="Numéro client",
                                help="le numéro client est composé de 6 chiffres",
                                placeholder="Entrez le numéro client ...",
                                max_chars=6,
                                label_visibility="collapsed")

# Si le client sélectionné a changé, réinitialiser la prédiction
if selected_client and selected_client != previous_client_id:
    if "prediction" in st.session_state:
        del st.session_state["prediction"]  # Supprimer la prédiction existante
    st.session_state.client_id = selected_client  # Mettre à jour le client_id
    
# vérification du client
if selected_client:
    if selected_client.isdigit() and len(selected_client) == 6:
        client_id_int = int(selected_client)
        if client_id_int in df_sample.index:
            st.session_state.client_id = selected_client
            if "prediction" not in st.session_state:
                # avertir l'utilisateur si la prédiction n'a pas été faite
                st.warning(f"Client {selected_client} sélectionné, veuillez vous diriger sur la page de prédiction.")
            else :
                st.success(f"Client sélectionné : {selected_client}")
        else:
            st.error("Ce numéro client n'existe pas dans les données.")
    else:
        st.error("Le numéro client doit contenir exactement 6 chiffres.")

st.page_link("pages/1_Prédiction.py", label="**Prédiction** : Obtenez une prédiction pour un client donné.")
st.page_link("pages/2_Explicabilité.py", label="**Explicabilité** : Comprenez pourquoi un client a une certaine prédiction.")
st.page_link("pages/3_Analyse.py", label="**Analyse** : Explorez les données et identifiez des tendances.")

# Afficher un client au hasard
client_au_hasard = df_sample.sample(1, random_state=None).index[0]
st.write(f"Exemple de client : {client_au_hasard}")

with st.expander("ou voir quelques exemples de numéros client disponibles"):
    st.write("Voici quelques exemples de clients présents dans la base :")
    exemples_clients = df_sample.sample(5, random_state=None).index.tolist()
    for client in exemples_clients:
        st.write(client)

# Lien vers la sidebar (géré automatiquement par Streamlit avec le dossier `pages/`)
st.sidebar.success("Renseignez un numéro client sur la page d'accueil puis sélectionnez une page dans le menu.")

st.sidebar.image("16794938722698_Data Scientist-P7-01-banner.png",
                 caption="Illustration : société Prêt à dépenser",
                 use_container_width=True)

# accessibilité taille du texte
taille_texte = st.sidebar.radio("Taille du texte", ["Normal", "Grand"])
if taille_texte == "Grand":
    st.markdown("<style>html, body, [class*='css'] { font-size: 18px !important; }</style>", unsafe_allow_html=True)
