import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap
# Interface Streamlit
st.title(
    "Explicabilité de la prédiction",
    help="""Cette page permet de comprendre pourquoi le modèle a pris sa décision. 
    L’onglet "Explicabilité globale" montre les variables qui influencent le plus le modèle en général, 
    et l’onglet "Explicabilité locale" explique la décision pour le client sélectionné."""
)

st.sidebar.image("16794938722698_Data Scientist-P7-01-banner.png",
                 caption="Illustration : société Prêt à dépenser",
                 use_container_width=True)

# accessibilité taille du texte
taille_texte = st.sidebar.radio("Taille du texte", ["Normal", "Grand"])
if taille_texte == "Grand":
    st.markdown("<style>html, body, [class*='css'] { font-size: 18px !important; }</style>", unsafe_allow_html=True)
client_id = st.session_state.get("client_id", None)

# Vérification que client_id existe, n'est pas vide et est un nombre entier
if not client_id or not client_id.isdigit():
    st.warning("Veuillez d'abord sélectionner un client sur la page d'accueil.")
    if st.button("Retour à l'accueil ..."):
        st.switch_page("Accueil.py")
    st.stop()

client_id_int = int(client_id)

# charger les données
df_sample = pd.read_csv('df_sample.csv')
# définir SK_ID_CUR en index
df_sample = df_sample.set_index('SK_ID_CURR')
# charger le scaler
scaler = joblib.load('scaler.pkl')

# charger l'explaineur shap
shap_explainer = joblib.load('global_explainer.pkl')

X = df_sample.drop(columns=['TARGET'])

X_scaled = scaler.transform(X)
# definir un df avec X_scaled
X_scaled_df = pd.DataFrame(X_scaled,
                           index=X.index,
                           columns=X.columns)

# calculer les valeurs shap pour toutes les observations
shap_values = shap_explainer(X_scaled_df)

tab_1, tab_2 = st.tabs(["Explicabilité globale", "Explicabilité locale"])
with tab_1:
    st.header("explicabilité globale",
              help="L'explicabilité globale permet de comprendre quelles variables ont influencé le modèle dans ses prédictions, en montrant leur impact global sur l'ensemble des données")
    max_features_global = st.selectbox(label="Sélectionner le nombre de variables à afficher (par défaut 10)",
                options=[5, 10, 15, 20, 25, 30],
                        index=1,
                        help="Le graphique SHAP summary plot montre l'importance de chaque variable sur les prédictions du modèle. Chaque point représente un crédit. - **Axe Y** : Liste des features les plus influentes, classées par ordre d'importance moyenne (de haut en bas). - **Axe X** : Valeurs SHAP, indiquant l’impact sur la prédiction. - **Couleurs** : Le rouge signifie une valeur élevée de la feature, le bleu une valeur basse. - **Dispersion des points** : Montre la variabilité de l’impact de la feature sur différentes observations.")
    fig, ax = plt.subplots()
    # visualisation de la feature importance globale
    shap.summary_plot(shap_values,
                    X_scaled,
                    feature_names=X.columns,
                    max_display=max_features_global,
                    show=False)
    st.pyplot(fig)

    with st.expander("Comment lire ce graphique ?"):
        st.write(
            """Le graphique SHAP summary plot montre l'importance de chaque variable sur les prédictions du modèle. Chaque point représente un crédit.
            - **Axe Y** : Liste des features les plus influentes, classées par ordre d'importance moyenne (de haut en bas).
            - **Axe X** : Valeurs SHAP, indiquant l’impact sur la prédiction. Les valeurs négatives ont une importance sur la prédiction que le client n'aura pas d'incident de paiement. les valeurs positives ont une importance sur la prédiction que le client aura un  incident de paiement.
            - **Couleurs** : Le rouge signifie une valeur élevée de la feature, le bleu une valeur basse.
            - **Dispersion des points** : Montre la variabilité de l’impact de la feature sur différentes observations."""
        )
with tab_2:
    st.header("explicabilité locale",
              help="L'explicabilité locale permet de comprendre pourquoi le modèle a pris une décision spécifique pour un client donné, en analysant l'impact des variables sur sa prédiction.")
    max_features_local = st.select_slider(label="Sélectionner le nombre de variables à afficher (par défaut 10)",
                                          options=list(range(5, 31)),
                                          value=10,
                                          help="Le graphique SHAP waterfall explique la prédiction du modèle pour un client spécifique : - **Axe Y** : Les variables les plus influentes, classées par ordre d'importance (de haut en bas). - **Axe X** : Les valeurs SHAP, montrant l'effet de chaque variable sur la prédiction finale. - **Couleurs** : Le rouge signifie que la variable augmente le risque de défaut, le bleu signifie que la variable diminue le risque de défaut. Les valeurs en gris indiquent les valeurs réelles prises par les variables pour ce client. - **Comment lire une SHAP value** : Le modèle commence avec une prédiction de base (la moyenne des prédictions sur tous les clients). Ensuite, il ajoute ou soustrait les contributions de chaque variable (les SHAP values), jusqu'à arriver à la prédiction finale pour le client. Par exemple, si la moyenne est 0.5 et que certaines variables ajoutent +0.1 et d'autres -0.05, la prédiction finale pourrait être 0.55. Ce graphique permet ainsi de visualiser comment chaque variable a modifié la prédiction pour ce client, et donc d'expliquer la décision du modèle.")
            
    if client_id_int in X_scaled_df.index:
        index_client = X_scaled_df.index.get_loc(client_id_int)
        st.write(f"Client {client_id_int} :")
        fig_2, ax_2 = plt.subplots()
        shap.waterfall_plot(shap_values[index_client],
                            max_display=max_features_local,
                            show=False)
        st.pyplot(fig_2)
        with st.expander("Comment lire ce graphique ?"):
            st.write(
                """Le graphique SHAP waterfall explique la prédiction du modèle pour un client spécifique :
                - **Axe Y** : Les variables les plus influentes, classées par ordre d'importance (de haut en bas).
                - **Axe X** : Les valeurs SHAP, montrant l'effet de chaque variable sur la prédiction finale.
                - **Couleurs** :
                    - Rouge : La variable augmente le risque de défaut.
                    - Bleu : La variable diminue le risque de défaut.
                    - Les valeurs en gris indiquent les valeurs réelles prises par les variables pour ce client.
                - **Comment lire une SHAP value** :
                Le modèle commence avec une prédiction de base (la moyenne des prédictions sur tous les clients). Ensuite, il ajoute ou soustrait les contributions de chaque variable (les SHAP values), jusqu'à arriver à la prédiction finale pour le client.
                Par exemple, si la moyenne est 0.5 et que certaines variables ajoutent +0.1 et d'autres -0.05, la prédiction finale pourrait être 0.55.
                Ce graphique permet ainsi de visualiser comment chaque variable a modifié la prédiction pour ce client, et donc d'expliquer la décision du modèle."""
            )
    else:
        st.error(f"Client {client_id_int} non trouvé dans les données")
        st.stop()

if st.button("retour à l'accueil"):
    st.switch_page("Accueil.py")
if st.button("aller à l'analyse"):
    st.switch_page("pages/3_Analyse.py")