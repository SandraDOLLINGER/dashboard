import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.title(
    "Analyse du client",
    help="Cette page permet de voir comment le client se positionne par rapport aux autres. Le premier onglet donne une vue d'ensemble, les autres onglets permettent de comparer ses informations avec celles d'autres clients à l’aide de graphiques simples : comparaisons à deux variables, diagrammes en boîte et courbes de répartition."
)

st.sidebar.image("16794938722698_Data Scientist-P7-01-banner.png",
                 caption="Illustration : société Prêt à dépenser",
                 use_container_width=True)

# accessibilité taille du texte
taille_texte = st.sidebar.radio("Taille du texte", ["Normal", "Grand"])
if taille_texte == "Grand":
    st.markdown("<style>html, body, [class*='css'] { font-size: 18px !important; }</style>", unsafe_allow_html=True)
client_id = st.session_state.get("client_id", None)
prediction_client = st.session_state.get("prediction", None)

# Vérification que client_id existe, n'est pas vide et est un nombre entier
if not client_id or not client_id.isdigit():
    st.warning("Veuillez d'abord sélectionner un client sur la page d'accueil.")
    if st.button("Retour à l'accueil ..."):
        st.switch_page("Accueil.py")
    st.stop()

# Vérification que la prédiction est faite
if prediction_client is None:
    st.warning("La prédiction n'est pas disponible. Veuillez passer par la page Prédiction.")
    if st.button("Aller à la page Prédiction ..."):
        st.switch_page("pages/1_Prédiction.py")
    st.stop()

client_id_int = int(client_id)

# définition des couleurs
color_non_def = "#404040"
color_def = "#0F60B6"
color_client = "#FF00FF"

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

tab_a, tab_b, tab_c, tab_d = st.tabs(["Comparaison du client aux autres clients", "Analyse bivariée du client", "Boxplot", "KDE"])
with tab_a:
    st.header("Graphiques comparatifs",
             help="Ce graphique montre où se situe la variable choisie pour le client par rapport aux autres clients. - Barres anthracites : Représentent la distribution des clients sans incidents de paiement. - Barres bleues rayées : Représentent la distribution des clients avec incidents de paiement. - Ligne verticale magenta : Indique la valeur de la variable choisie pour le client sélectionné. Cela vous aide à comprendre si la valeur de cette variable place le client dans une position similaire à celle des autres clients avec ou sans incidents.",
             divider="gray")
    variable_choisie = st.selectbox(label="Variable à comparer",
                                    options=df_streamlit_filtered.columns,
                                    index=None,
                                    placeholder="Choississez une variable ...")
    if variable_choisie:
        df_plot = df_streamlit_filtered.copy()
        df_plot["TARGET"] = df_pour_streamlit["TARGET"]
        client_value = df_streamlit_filtered.loc[client_id_int, variable_choisie]
        
        # Préparation des histogrammes "manuels"
        data_non_def = df_plot[df_plot["TARGET"] == 0][variable_choisie]
        data_def = df_plot[df_plot["TARGET"] == 1][variable_choisie]

        # Définir les bornes des bins
        bins = np.histogram_bin_edges(df_plot[variable_choisie], bins=30)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Compter les valeurs dans chaque bin
        hist_non_def, _ = np.histogram(data_non_def, bins=bins)
        hist_def, _ = np.histogram(data_def, bins=bins)

        fig = go.Figure()

        # Clients sans incidents (barres pleines classiques)
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist_non_def,
            name="Clients sans incidents de paiement",
            marker=dict(color=color_non_def),
            width=np.diff(bins)
        ))

        # Clients avec incidents (hachures simulées)
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist_def,
            name="Clients avec incident de paiement",
            marker=dict(
                color=color_def,
                pattern=dict(
                    shape="\\",  # motif hachuré
                    fillmode="overlay",
                    size=10,
                    solidity=0.2
                )
            ),
            width=np.diff(bins)
        ))

        # Ligne verticale pour le client
        fig.add_vline(
            x=client_value,
            line_dash="dash",
            line_color=color_client,
            line_width=5,
            annotation_text=f"{variable_choisie} du client : {client_value:.2f}",
            annotation_position="top right",
            annotation_font=dict(color=color_client),
        )

        fig.update_layout(
            barmode='overlay',
            title=f"Distribution de {variable_choisie} selon la TARGET",
            xaxis_title=variable_choisie,
            yaxis_title="Fréquence",
            legend=dict(x=0.5, y=1.15, orientation="h", xanchor="center"),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez sélectionner une variable pour afficher la comparaison.")

with tab_b:
    st.header("Analyse bivariée du client",
             help="L'analyse bivariée examine la relation entre deux variables, en l'occurrence les variables X (en abscisse) et Y (en ordonnée). Les clients sont classés en fonction de leur statut de paiement : - Clients sans incident de paiement sont représentés par des points anthracite. - Clients avec incident de paiement sont représentés par des carrés bleus. - Le client en question est mis en avant avec une croix magenta. Cela vous permet de visualiser facilement où se situe le client par rapport aux autres et de comparer ses valeurs avec celles des autres clients ayant des comportements de paiement différents.",
                divider="gray")

    # Sélection des variables à comparer en bivarié
    x_var = st.selectbox(label="Variable à afficher en abscisse (x) :",
                         options=df_streamlit_filtered.columns,
                         index=None,
                         placeholder="Choisissez une variable ...",
                         key="x_var")

    y_var = st.selectbox(label="Variable à afficher en ordonnée (y) :",
                         options=df_streamlit_filtered.columns,
                         index=None,
                         placeholder="Choisissez une variable ...",
                         key="y_var")

    if x_var and y_var:
        df_plot = df_streamlit_filtered.copy()
        df_plot["TARGET"] = df_pour_streamlit["TARGET"]

        # Création du graphique de dispersion bivarié avec des formes différentes
        fig = go.Figure()

        # Clients sans incident : points ronds
        non_def_clients = df_plot[df_plot["TARGET"] == 0]
        fig.add_trace(go.Scatter(
            x=non_def_clients[x_var],
            y=non_def_clients[y_var],
            mode='markers',
            marker=dict(symbol='circle', color=color_non_def, size=6),
            name="Clients sans incident de paiement"
        ))

        # Clients avec incident de paiement : points carrés
        def_clients = df_plot[df_plot["TARGET"] == 1]
        fig.add_trace(go.Scatter(
            x=def_clients[x_var],
            y=def_clients[y_var],
            mode='markers',
            marker=dict(symbol='square', color=color_def, size=10),
            name="Clients avec incident de paiement"
        ))

        # Ajout du point du client sélectionné : point en forme de croix
        client_value_x = df_streamlit_filtered.loc[client_id_int, x_var]
        client_value_y = df_streamlit_filtered.loc[client_id_int, y_var]

        fig.add_trace(go.Scatter(
            x=[client_value_x],
            y=[client_value_y],
            mode='markers',
            marker=dict(symbol='x', color=color_client, size=16),
            name=f"Client {client_id_int}"
        ))

        # Titre et axes du graphique
        fig.update_layout(
            title=f"Relation entre {x_var} et {y_var} selon la TARGET",
            xaxis_title=x_var,
            yaxis_title=y_var,
            legend_title="Classe de client",
            plot_bgcolor='white'
        )

        st.plotly_chart(fig)
    else:
        st.info("Veuillez sélectionner deux variables pour afficher le graphique.")

with tab_c:
    st.header("Graphiques comparatifs - version Boxplot",
              help="Le graphique Boxplot montre la distribution de la variable choisie entre les clients avec incident de paiement et sans incident de paiement. - Clients sans incident de paiement : représentés par la boîte anthracite. - Clients avec incident de paiement : représentés par la boîte bleue. - Le client en question : indiqué par une croix magenta. Ce graphique permet de visualiser la dispersion des valeurs pour chaque groupe, et où se situe le client par rapport à ces deux groupes en termes de la variable choisie.",
                divider="gray")
    variable_choisie = st.selectbox(label="Variable à comparer (boxplot)",
                                    options=df_streamlit_filtered.columns,
                                    index=None,
                                    placeholder="Choisissez une variable ...",
                                    key="boxplot_select")
    if variable_choisie:
        df_plot = df_streamlit_filtered.copy()
        df_plot["TARGET"] = df_pour_streamlit["TARGET"]
        client_value = df_streamlit_filtered.loc[client_id_int, variable_choisie]

        fig = go.Figure()

        # Clients sans incidents
        fig.add_trace(go.Box(
            y=df_plot[df_plot["TARGET"] == 0][variable_choisie],
            name="Clients sans incidents de paiement",
            marker=dict(color=color_non_def),
            boxpoints='all',
            line_color=color_non_def
        ))

        # Clients avec incidents
        fig.add_trace(go.Box(
            y=df_plot[df_plot["TARGET"] == 1][variable_choisie],
            name="Clients avec incidents de paiement",
            marker=dict(color=color_def),
            boxpoints='all',
            line_color=color_def
        ))

        # Ajout du point client
        fig.add_trace(go.Scatter(
            # x=["Clients avec incidents de paiement" if df_pour_streamlit.loc[client_id_int, "TARGET"] == 1 else "Clients sans incidents de paiement"],
            x=["Clients avec incidents de paiement" if prediction_client == 1 else "Clients sans incidents de paiement"],
            y=[df_streamlit_filtered.loc[client_id_int, variable_choisie]],
            mode='markers',
            marker=dict(color=color_client, size=16, symbol="x"),
            name=f"Client {client_id_int}"
        ))

        fig.update_layout(
            title=f"{variable_choisie} selon la TARGET (boxplot)",
            xaxis_title="Cible (TARGET)",
            yaxis_title=variable_choisie,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig)

with tab_d:
    st.header("Graphiques comparatifs - version KDEplot",
              help="Ce graphique de densité permet de comparer la distribution d'une variable continue entre les clients ayant eu un incident de paiement (en bleu) et ceux n'en ayant pas eu (en anthracite). La ligne en pointillés magenta représente la position du client analysé sur cette variable. Cela permet de visualiser rapidement si ce client se situe dans une zone à risque ou non, en fonction des tendances observées dans l'ensemble des données.",
                divider="gray")
    variable_choisie = st.selectbox(label="Variable à comparer (kdeplot)",
                                    options=df_streamlit_filtered.columns,
                                    index=None,
                                    placeholder="Choisissez une variable ...",
                                    key="kdeplot_select")

    if variable_choisie:
        df_plot = df_streamlit_filtered.copy()
        df_plot["TARGET"] = df_pour_streamlit["TARGET"]
        client_value = df_streamlit_filtered.loc[client_id_int, variable_choisie]

        # Séparer les groupes
        values_0 = df_plot[df_plot["TARGET"] == 0][variable_choisie].dropna()
        values_1 = df_plot[df_plot["TARGET"] == 1][variable_choisie].dropna()

        # Créer un axe X commun
        x_grid = np.linspace(min(df_plot[variable_choisie]), max(df_plot[variable_choisie]), 500)

        # KDE pour chaque groupe
        kde_0 = gaussian_kde(values_0)(x_grid)
        kde_1 = gaussian_kde(values_1)(x_grid)

        # Construction du graphique
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_grid,
            y=kde_0,
            fill='tozeroy',
            mode='lines',
            line=dict(color=color_non_def),
            name="Clients sans incident de paiement"
        ))

        fig.add_trace(go.Scatter(
            x=x_grid,
            y=kde_1,
            fill='tozeroy',
            mode='lines',
            line=dict(color=color_def, dash="dot", width=3),
            name="Clients avec incident de paiement"
        ))

        fig.add_trace(go.Scatter(
            x=[client_value, client_value],
            y=[0, max(kde_0.max(), kde_1.max()) * 1.05],
            mode='lines',
            line=dict(color=color_client, width=5, dash="dash"),
            name=f"Client {client_id_int}"
        ))

        fig.update_layout(
            title=f"Densité estimée de {variable_choisie} selon la TARGET",
            xaxis_title=variable_choisie,
            yaxis_title="Densité",
        )

        st.plotly_chart(fig)

if st.button("retour à l'accueil"):
    st.switch_page("Accueil.py")