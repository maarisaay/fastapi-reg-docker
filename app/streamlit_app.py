import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8008")

st.set_page_config(
    page_title="Linear Regression ML App",
    page_icon="📈",
    layout="centered"
)

st.title("Linear Regression ML App")
st.write(
    "Aplikacja umożliwia predykcję wartości `y` dla podanego `x`, "
    "dotrenowanie modelu nowym punktem oraz wizualizację danych treningowych"
)


st.header("Predykcja wartości y")

x_value = st.number_input("Podaj wartość x:", value=1.0)

if st.button("Przewidź y"):
    try:
        response = requests.post(
            f"{API_URL}/model/predict",
            json={"x": x_value}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Dla x = {result['x']} przewidywane y = {result['y']:.4f}")
        else:
            st.error(response.json())

    except requests.exceptions.RequestException as e:
        st.error("Nie udało się połączyć z backendem.")
        st.write(e)


st.header("Dotrenowanie modelu")

new_x = st.number_input("Nowe x:", value=1.0, key="new_x")
new_y = st.number_input("Nowe y:", value=1.0, key="new_y")

if st.button("Dodaj punkt i dotrenuj model"):
    try:
        response = requests.post(
            f"{API_URL}/model/train",
            json={
                "x": new_x,
                "y": new_y
            }
        )

        if response.status_code == 200:
            st.success("Model został zaktualizowany.")
        else:
            st.error(response.json())

    except requests.exceptions.RequestException as e:
        st.error("Nie udało się połączyć z backendem.")
        st.write(e)


st.header("Dane treningowe i linia regresji")

try:
    data_response = requests.get(f"{API_URL}/model/data")
    params_response = requests.get(f"{API_URL}/model/params")

    if data_response.status_code == 200 and params_response.status_code == 200:
        data = data_response.json()
        params = params_response.json()

        df = pd.DataFrame(data)

        st.subheader("Tabela danych")
        st.dataframe(df)

        coef = params["coef"]
        intercept = params["intercept"]

        st.subheader("Parametry modelu")
        st.write(f"Współczynnik kierunkowy a: `{coef:.4f}`")
        st.write(f"Wyraz wolny b: `{intercept:.4f}`")

        st.subheader("Wykres punktowy z linią regresji")

        fig, ax = plt.subplots()

        ax.scatter(df["x"], df["y"], label="Dane treningowe")

        x_line = df["x"].sort_values()
        y_line = coef * x_line + intercept

        ax.plot(x_line, y_line, label="Linia regresji")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        st.pyplot(fig)

    else:
        st.error("Nie udało się pobrać danych lub parametrów modelu.")

except requests.exceptions.RequestException as e:
    st.error("Nie udało się połączyć z backendem.")
    st.write(e)