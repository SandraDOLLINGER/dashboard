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
# Interface Streamlit
st.title("Explicabilité de la prédiction")
st.sidebar.image("16794938722698_Data Scientist-P7-01-banner.png")
client_id = st.session_state.get("client_id", None)

# Vérification que client_id existe, n'est pas vide et est un nombre entier
if not client_id or not client_id.isdigit():
    st.warning("Veuillez d'abord sélectionner un client sur la page d'accueil.")
    if st.button("Retour à l'accueil ..."):
        st.switch_page("Accueil.py")
    st.stop()

client_id_int = int(client_id)

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





# col1, col2 = st.columns(2)
# with col1:
#         # Fonction pour récupérer la prédiction de l'API FastAPI
#         def get_prediction(client_id):
#             try:
#                 # Faire une requête GET à l'API
#                 response = requests.get(f"{API_URL}/{client_id}")
#                 if response.status_code == 200:
#                     data = response.json()
#                     return data
#                 elif response.status_code == 404:
#                     st.error("Client non trouvé, renseignez un numéro client")
#                     return None
#                 else:
#                     st.error("Erreur dans la requête à l'API")
#                     return None
#             except Exception as e:
#                 st.error(f"Erreur : {e}")
#                 return None
#         if client_id:
#             # vérifier si le client_id est valide
#             if not client_id.isdigit():
#                 st.error("Le numéro client doit être un nombre entier, renseignez un numéro de client")
#             elif len(client_id) != 6:
#                 st.error("Le numéro client doit avoir exactement 6 chiffres, renseignez un numéro de client")
#             else:
#                 # convertir en entier
#                 client_id_int = int(client_id)    
#                 # Appeler la fonction pour obtenir la prédiction
#                 result = get_prediction(client_id_int)
            
#                 if result is not None:
#                     prediction_classe = result["prediction"]
#                     probability = result["probability"]

#                     st.write(f"### Client {client_id}")
#         #            st.write(f'Prédiction : {prediction_classe}')
#                     st.write(f"**Prédiction :** {'Refusé' if prediction_classe == 1 else 'Accepté'}")
#                     st.write(f"**Probabilité de refus :** {probability:.2%}")     
#         else:
#             st.write("Veuillez entrer un numéro de client.")

# with col2:
#     if probability is not None:
#         streamviz.gauge(
#             gVal=probability,
#             gTitle="Probabilité d'incident de paiement",
#             sFix="%",
#             gSize="SML",
#             gcLow= "palegreen",
#             gcMid="navajowhite",
#             gcHigh="darksalmon",
#             grLow=meilleur_seuil_value-0.1,
#             grMid=meilleur_seuil_value+0.1)
# st.write("-" * 180)
# # Define the gauge chart
# fig = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=probability,
#     title={'text': "Probabilité d'incident de paiement du client"},
#     gauge={'axis': {'range': [0.0, 1.0]},
#         'bar': {'color': "darkblue"},
#         'steps': [
#             {'range': [0.0, (meilleur_seuil_value - 0.1)], 'color': "palegreen"},
#             {'range': [(meilleur_seuil_value - 0.1), (meilleur_seuil_value + 0.1)], 'color': "navajowhite"},
#             {'range': [(meilleur_seuil_value + 0.1), 1.0], 'color': "darksalmon"}
#             ],
#         'threshold': {'line': {'color': "orangered", 'width': 4}, 'thickness': 1, 'value': meilleur_seuil_value}}))
# # Display the gauge chart in Streamlit
# st.plotly_chart(fig)


tab_1, tab_2 = st.tabs(["Explicabilité globale", "Explicabilité locale"])
with tab_1:
    st.header("explicabilité globale")
    max_features_global = st.selectbox(label="Sélectionner le nombre de variables à afficher (par défaut 10)",
                options=[5, 10, 15, 20, 25, 30],
                        index=1,
                        help="Le graphique SHAP summary plot montre l'importance de chaque variable sur les prédictions du modèle. Chaque point représente un crédit. - **Axe Y** : Liste des features les plus influentes, classées par importance moyenne. - **Axe X** : Valeurs SHAP, indiquant l’impact sur la prédiction. - **Couleurs** : Le rouge signifie une valeur élevée de la feature, le bleu une valeur basse. - **Dispersion des points** : Montre la variabilité de l’impact de la feature sur différentes observations.")
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
            - **Axe Y** : Liste des features les plus influentes, classées par importance moyenne.
            - **Axe X** : Valeurs SHAP, indiquant l’impact sur la prédiction.
            - **Couleurs** : Le rouge signifie une valeur élevée de la feature, le bleu une valeur basse.
            - **Dispersion des points** : Montre la variabilité de l’impact de la feature sur différentes observations."""
        )
with tab_2:
    st.header("explicabilité locale",
              help="Le graphique **SHAP waterfall** permet d'expliquer la prédiction du modèle pour un client donné. - **Axe Y** : les variables les plus influentes, classées de haut en bas par importance. - **Axe X** : les valeurs SHAP associées à chaque variable, indiquant leur contribution à la prédiction. - **Couleurs** : - **Rouge** : la variable contribue à augmenter le risque de défaut. - **Bleu** : la variable contribue à diminuer le risque. - Les valeurs affichées en gris indiquent les valeurs réelles prises par chaque variable pour ce client. Ce graphique aide à comprendre les raisons derrière la décision du modèle.")
    max_features_local = st.select_slider(label="Sélectionner le nombre de variables à afficher (par défaut 10)",
                                          options=list(range(5, 31)),
                                          value=10,
                                          help="Le graphique **SHAP waterfall** permet d'expliquer la prédiction du modèle pour un client donné. - **Axe Y** : les variables les plus influentes, classées de haut en bas par importance. - **Axe X** : les valeurs SHAP associées à chaque variable, indiquant leur contribution à la prédiction. - **Couleurs** : - **Rouge** : la variable contribue à augmenter le risque de défaut. - **Bleu** : la variable contribue à diminuer le risque. - Les valeurs affichées en gris indiquent les valeurs réelles prises par chaque variable pour ce client. Ce graphique aide à comprendre les raisons derrière la décision du modèle.")
            
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
                """Le graphique **SHAP waterfall** permet d'expliquer la prédiction du modèle pour un client donné.

        - **Axe Y** : les variables les plus influentes, classées de haut en bas par importance.
        - **Axe X** : les valeurs SHAP associées à chaque variable, indiquant leur contribution à la prédiction.
        - **Couleurs** :
            - **Rouge** : la variable contribue à augmenter le risque de défaut.
            - **Bleu** : la variable contribue à diminuer le risque.
        - Les valeurs affichées en gris indiquent les valeurs réelles prises par chaque variable pour ce client.

        Ce graphique aide à comprendre les raisons derrière la décision du modèle."""
            )
    else:
        st.error(f"Client {client_id_int} non trouvé dans les données")
        st.stop()

# tab_a, tab_b = st.tabs(["Comparaison du client aux autres clients", "Analyse bivariée du client"])
# with tab_a:
#     st.write("Graphiques comparatifs")
#     variable_choisie = st.selectbox(label="Choisir une variable à comparer",
#                  options=df_streamlit_filtered.columns)
#     client_value = df_streamlit_filtered.loc[client_id_int, variable_choisie]
#     fig, ax = plt.subplots()
#     ax.hist(df_streamlit_filtered[variable_choisie],
#             bins=30,
#             color='royalblue',
#             edgecolor='black',
#             alpha=0.7)
#     ax.axvline(client_value,
#                color='orange',
#                linestyle='dashed',
#                linewidth=2,
#                label=f"{variable_choisie} du client")
#     ax.legend()
#     ax.set_xlabel(variable_choisie)
#     ax.set_ylabel("Fréquence")
#     ax.set_title(f"Distribution de {variable_choisie}")
#     st.pyplot(fig)
#     st.write(variable_choisie)
    
# with tab_b:
#     st.write("Analyse bivariée")
#     x_var = st.selectbox("Choissisez la variable à afficher en abscisse (x) :",
#                          df_streamlit_filtered.columns)
#     y_var = st.selectbox("Choissisez la variable à afficher en abscisse (y) :",
#                          df_streamlit_filtered.columns)
#     if x_var and y_var:
#         fig_4 = px.scatter(df_streamlit_filtered,
#                          x=x_var,
#                          y=y_var,
#                          title= f"Scatter plot : {x_var} vs {y_var}")
#         client_value_x = df_streamlit_filtered.loc[client_id_int, x_var]
#         client_value_y = df_streamlit_filtered.loc[client_id_int, y_var]
#         fig_4.add_trace(go.Scatter(x=[client_value_x],
#                              y=[client_value_y],
#                              mode='markers',
#                              marker=dict(color='orange', size=12, symbol='x'),
#                              name=f"Client {client_id_int}"))
#         st.plotly_chart(fig_4)

if st.button("retour à l'accueil"):
    st.switch_page("Accueil.py")
if st.button("aller à l'analyse"):
    st.switch_page("pages/3_Analyse.py")