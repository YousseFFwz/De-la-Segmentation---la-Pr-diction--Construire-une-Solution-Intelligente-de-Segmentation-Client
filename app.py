import streamlit as st
import pandas as pd
import joblib


rf_model = joblib.load("notebooks/random_forest_model.joblib")
st.title("Customer Segmentation App")
recency = st.number_input("Recency", min_value=0, value=20)
frequency = st.number_input("Frequency", min_value=1, value=10)
monetary = st.number_input("Monetary", min_value=0.0, value=5000.0)
if st.button("Prédire le segment"):
    new_client = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    predicted_cluster = rf_model.predict(new_client)[0]

    cluster_names = {
        0: "Clients à forte valeur",
        1: "Clients peu actifs",
        2: "Clients intermédiaires"
    }

    predicted_name = cluster_names[predicted_cluster]
    st.subheader("Résultat de la prédiction")
    st.write("Cluster :", predicted_cluster)
    st.success(f"Segment : {predicted_name}")