import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Variables independientes del modelo
FEATURES = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3',
           'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']


def load_model():
    try:
        # Ruta específica a tu modelo
        model_path = 'modelo_titanic.joblib'        
        if not os.path.exists(model_path):
            st.error(f"No se encontró el modelo en: {model_path}")
            st.info("Por favor, asegúrate de que el archivo 'rf_model.joblib' esté en el mismo directorio que esta aplicación.")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

model = load_model()

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create input data
    data = {
        'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
        'Pclass_1': 1 if pclass == 1 else 0,
        'Pclass_2': 1 if pclass == 2 else 0,
        'Pclass_3': 1 if pclass == 3 else 0,
        'Sex_female': 1 if sex == 'female' else 0,
        'Sex_male': 1 if sex == 'male' else 0,
        'Embarked_C': 1 if embarked == 'C' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0
    }

    df = pd.DataFrame([data])[FEATURES]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return "Survived" if prediction == 1 else "Did not survive", f"{probability*100:.1f}%"


col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Información Personal")

    # Edad
    age = st.slider("Edad", min_value=0, max_value=100, value=30, step=1)

    # Sexo
    sex = st.selectbox("Sexo", options=['male', 'female'],
                      format_func=lambda x: 'Masculino' if x == 'male' else 'Femenino')

    # Clase del pasajero
    pclass = st.selectbox("Clase del Pasajero", options=[1, 2, 3],
                         format_func=lambda x: f"Clase {x} ({'Primera' if x==1 else 'Segunda' if x==2 else 'Tercera'})")

with col2:
    st.subheader("👨‍👩‍👧‍👦 Información Familiar y Viaje")

    # Hermanos/cónyuges a bordo
    sibsp = st.number_input("Hermanos/Cónyuges a bordo", min_value=0, max_value=10, value=0, step=1)

    # Padres/hijos a bordo
    parch = st.number_input("Padres/Hijos a bordo", min_value=0, max_value=10, value=0, step=1)

    # Tarifa
    fare = st.number_input("Tarifa pagada (£)", min_value=0.0, max_value=1000.0, value=50.0, step=0.1)

    # Puerto de embarque
    embarked_options = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    embarked = st.selectbox("Puerto de Embarque",
                           options=['C', 'Q', 'S'],
                           format_func=lambda x: embarked_options[x])

# Botón de predicción
st.markdown("---")

if st.button("🔮 Predecir Supervivencia", type="primary", use_container_width=True):
    with st.spinner("Realizando predicción..."):
        result, probability = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)

        # Mostrar resultados
        st.markdown("### 📊 Resultado de la Predicción")

        col1, col2 = st.columns(2)

        with col1:
            if "Sobrevivió" in result:
                st.success(f"**Predicción:** {result}")
            else:
                st.error(f"**Predicción:** {result}")

        with col2:
            st.info(f"**Probabilidad de Supervivencia:** {probability}")

        # Barra de progreso visual
        prob_value = float(probability.replace('%', '')) / 100
        st.progress(prob_value)

        # Interpretación
        st.markdown("### 💭 Interpretación")
        if prob_value > 0.7:
            st.success("Alta probabilidad de supervivencia - Las características del pasajero son favorables.")
        elif prob_value > 0.4:
            st.warning("Probabilidad moderada de supervivencia - Resultado incierto.")
        else:
            st.error("Baja probabilidad de supervivencia - Las características del pasajero no son favorables.")
