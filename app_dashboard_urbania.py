# app_dashboard_urbania.py
# Dashboard predictivo para precios inmobiliarios con Streamlit
# Variable dependiente: ln(precio por m²)

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Dashboard inmobiliario - Urbania",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Dashboard predictivo de precios inmobiliarios")
st.caption("Modelo de regresión lineal usando ln(precio por m²) como variable dependiente")

DEFAULT_CSV = "urbania_base_limpia_modelo_final.csv"
OUTLIER_INDEX_R = [2, 3, 68, 162, 167, 580, 703, 721, 776, 801, 980, 1041, 1065, 1066, 1075]

FEATURES_NUM = ["ln_area_total_m2", "ratio_construccion", "banios", "cocheras"]
FEATURES_CAT = ["estado_inmueble", "tipo_vivienda", "distrito", "terraza"]
TARGET = "ln_precio_m2"


def make_encoder():
    """Compatible con versiones nuevas y antiguas de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif os.path.exists(DEFAULT_CSV):
        data = pd.read_csv(DEFAULT_CSV)
    else:
        return None

    # Asegurar variables necesarias
    if "ln_area_total_m2" not in data.columns and "area_total_m2" in data.columns:
        data["ln_area_total_m2"] = np.log(data["area_total_m2"])
    if "ratio_construccion" not in data.columns and {"area_constr_m2", "area_total_m2"}.issubset(data.columns):
        data["ratio_construccion"] = data["area_constr_m2"] / data["area_total_m2"]
    if TARGET not in data.columns and "precio_m2" in data.columns:
        data[TARGET] = np.log(data["precio_m2"])

    # Variables categóricas como texto
    for col in FEATURES_CAT:
        if col in data.columns:
            data[col] = data[col].astype(str)

    return data


def clean_outliers(data):
    # Índices detectados en el análisis R; R usa índice desde 1, Python desde 0.
    drop_pos = [i - 1 for i in OUTLIER_INDEX_R if 0 <= i - 1 < len(data)]
    return data.drop(index=data.index[drop_pos]).reset_index(drop=True)


def train_model(data):
    model_data = data.dropna(subset=FEATURES_NUM + FEATURES_CAT + [TARGET]).copy()

    X = model_data[FEATURES_NUM + FEATURES_CAT]
    y = model_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("cat", make_encoder(), FEATURES_CAT),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("reg", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    pred_price_m2 = np.exp(pred_log)
    real_price_m2 = np.exp(y_test)

    metrics = {
        "r2_log_test": r2_score(y_test, pred_log),
        "rmse_log_test": np.sqrt(mean_squared_error(y_test, pred_log)),
        "rmse_price_m2": np.sqrt(mean_squared_error(real_price_m2, pred_price_m2)),
        "mape_price_m2": mean_absolute_percentage_error(real_price_m2, pred_price_m2),
        "n_model": len(model_data),
    }

    return pipeline, metrics, model_data


def predict_property(model, area_total_m2, area_constr_m2, banios, cocheras,
                     estado_inmueble, tipo_vivienda, distrito, terraza):
    ratio = area_constr_m2 / area_total_m2 if area_total_m2 > 0 else 0
    ratio = min(max(ratio, 0), 3)

    input_df = pd.DataFrame([{
        "ln_area_total_m2": np.log(area_total_m2),
        "ratio_construccion": ratio,
        "banios": banios,
        "cocheras": cocheras,
        "estado_inmueble": str(estado_inmueble),
        "tipo_vivienda": str(tipo_vivienda),
        "distrito": str(distrito),
        "terraza": str(int(terraza)),
    }])

    pred_log = float(model.predict(input_df)[0])
    pred_price_m2 = float(np.exp(pred_log))
    pred_total = pred_price_m2 * area_total_m2
    return pred_log, pred_price_m2, pred_total, ratio


def format_money(x):
    return f"{x:,.0f}"


uploaded_file = st.sidebar.file_uploader("Sube tu CSV limpio", type=["csv"])
df = load_data(uploaded_file)

if df is None:
    st.warning("No se encontró el CSV. Sube tu archivo limpio en la barra lateral.")
    st.stop()

missing = [c for c in FEATURES_NUM + FEATURES_CAT + [TARGET, "precio_m2", "area_total_m2"] if c not in df.columns]
if missing:
    st.error(f"Faltan columnas necesarias en el dataset: {missing}")
    st.stop()

st.sidebar.header("Configuración del modelo")
n_remove = st.sidebar.checkbox(
    "Retirar 15 observaciones atípicas detectadas en R",
    value=True
)

if n_remove:
    data_model = clean_outliers(df)
else:
    data_model = df.copy()

model, metrics, model_data = train_model(data_model)

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Resumen", "📊 Exploración", "📈 Modelo", "🔮 Predicción"
])

with tab1:
    st.subheader("Resumen de la muestra")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", f"{len(data_model):,}")
    c2.metric("Precio/m² promedio", format_money(data_model["precio_m2"].mean()))
    c3.metric("Precio/m² mediano", format_money(data_model["precio_m2"].median()))
    c4.metric("Distritos", data_model["distrito"].nunique())

    st.markdown("""
    **Lectura rápida:** el modelo trabaja con `ln_precio_m2` porque permite reducir la dispersión del precio por m²
    y facilita interpretar los coeficientes como efectos porcentuales aproximados.
    """)

    st.dataframe(data_model.head(15), use_container_width=True)

with tab2:
    st.subheader("Análisis exploratorio")

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(data_model["ln_precio_m2"], bins=30)
        ax.set_title("Distribución del ln(precio/m²)")
        ax.set_xlabel("ln(precio/m²)")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    with c2:
        top_districts = (
            data_model.groupby("distrito", as_index=False)["precio_m2"]
            .median()
            .sort_values("precio_m2", ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(top_districts["distrito"], top_districts["precio_m2"])
        ax.set_title("Top 12 distritos por mediana de precio/m²")
        ax.set_ylabel("Precio/m²")
        ax.tick_params(axis="x", rotation=70)
        st.pyplot(fig)

    st.markdown("#### Correlación entre variables numéricas")
    corr_cols = ["ln_precio_m2", "ln_area_total_m2", "ratio_construccion", "banios", "cocheras"]
    corr = data_model[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Matriz de correlación")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Resultados del modelo predictivo")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² en test, escala log", f"{metrics['r2_log_test']:.3f}")
    c2.metric("RMSE log", f"{metrics['rmse_log_test']:.3f}")
    c3.metric("RMSE precio/m²", format_money(metrics["rmse_price_m2"]))
    c4.metric("MAPE precio/m²", f"{metrics['mape_price_m2']:.1%}")

    st.info(
        "El modelo predice primero el ln(precio/m²). Luego se aplica exp(predicción) "
        "para regresar a precio por m². El precio total estimado se obtiene multiplicando por el área total."
    )

    try:
        feature_names = model.named_steps["prep"].get_feature_names_out()
        coefs = model.named_steps["reg"].coef_
        coef_df = pd.DataFrame({"variable": feature_names, "coeficiente": coefs})
        coef_df["abs_coef"] = coef_df["coeficiente"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).head(20)
        st.markdown("#### Variables con mayor efecto en el modelo")
        st.dataframe(coef_df[["variable", "coeficiente"]], use_container_width=True)
    except Exception:
        st.write("No se pudieron mostrar los coeficientes transformados del modelo.")

with tab4:
    st.subheader("Simulador de predicción")
    st.caption("La predicción se expresa en la misma moneda de tu base de datos.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        area_total = st.number_input("Área total m²", min_value=20.0, max_value=10000.0, value=250.0, step=10.0)
        area_constr = st.number_input("Área construida m²", min_value=1.0, max_value=10000.0, value=220.0, step=10.0)
        banios = st.number_input("Baños", min_value=1, max_value=10, value=4, step=1)
        cocheras = st.number_input("Cocheras", min_value=0, max_value=10, value=2, step=1)
    with col_b:
        distrito = st.selectbox("Distrito", sorted(data_model["distrito"].dropna().astype(str).unique()), index=0)
        estado = st.selectbox("Estado del inmueble", sorted(data_model["estado_inmueble"].dropna().astype(str).unique()))
        tipo = st.selectbox("Tipo de vivienda", sorted(data_model["tipo_vivienda"].dropna().astype(str).unique()))
        terraza = st.selectbox("¿Tiene terraza?", [0, 1], index=1)
    with col_c:
        pred_log, pred_m2, pred_total, ratio = predict_property(
            model, area_total, area_constr, banios, cocheras,
            estado, tipo, distrito, terraza
        )
        st.metric("Predicción ln(precio/m²)", f"{pred_log:.3f}")
        st.metric("Precio estimado por m²", format_money(pred_m2))
        st.metric("Precio total estimado", format_money(pred_total))
        st.write(f"Ratio construcción: **{ratio:.2f}**")

    st.markdown("#### Ejemplos rápidos de predicción")
    examples = pd.DataFrame([
        {
            "caso": "Casa amplia en Cieneguilla",
            "area_total_m2": 1000,
            "area_constr_m2": 250,
            "banios": 3,
            "cocheras": 2,
            "estado_inmueble": "Bueno",
            "tipo_vivienda": "Casa",
            "distrito": "Cieneguilla",
            "terraza": 1,
        },
        {
            "caso": "Casa urbana en Chorrillos",
            "area_total_m2": 250,
            "area_constr_m2": 220,
            "banios": 4,
            "cocheras": 2,
            "estado_inmueble": "Muy bueno",
            "tipo_vivienda": "Casa",
            "distrito": "Chorrillos",
            "terraza": 1,
        },
        {
            "caso": "Casa premium en Miraflores",
            "area_total_m2": 300,
            "area_constr_m2": 270,
            "banios": 5,
            "cocheras": 3,
            "estado_inmueble": "Excelente",
            "tipo_vivienda": "Casa",
            "distrito": "Miraflores",
            "terraza": 1,
        },
        {
            "caso": "Casa en condominio en Ate",
            "area_total_m2": 180,
            "area_constr_m2": 170,
            "banios": 3,
            "cocheras": 1,
            "estado_inmueble": "Bueno",
            "tipo_vivienda": "Casa en condominio",
            "distrito": "Ate",
            "terraza": 0,
        },
    ])

    results = []
    for _, row in examples.iterrows():
        pred_log, pred_m2, pred_total, ratio = predict_property(
            model,
            row["area_total_m2"], row["area_constr_m2"],
            row["banios"], row["cocheras"],
            row["estado_inmueble"], row["tipo_vivienda"],
            row["distrito"], row["terraza"]
        )
        results.append({
            "caso": row["caso"],
            "distrito": row["distrito"],
            "area_total_m2": row["area_total_m2"],
            "ratio_construccion": round(ratio, 2),
            "precio_m2_estimado": round(pred_m2, 0),
            "precio_total_estimado": round(pred_total, 0),
        })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.warning(
        "Estas predicciones son referenciales. Funcionan mejor para inmuebles parecidos a los de la base usada para entrenar el modelo."
    )
