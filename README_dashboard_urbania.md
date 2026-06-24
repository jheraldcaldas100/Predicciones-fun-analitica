# Dashboard inmobiliario Urbania

Este proyecto usa el dataset limpio para entrenar un modelo de regresión lineal con `ln_precio_m2` como variable dependiente.

## Archivos

- `app_dashboard_urbania.py`: aplicación Streamlit.
- `urbania_base_limpia_modelo_final.csv`: dataset limpio.
- `requirements_dashboard_urbania.txt`: librerías necesarias.

## Cómo ejecutarlo

1. Crea y activa un entorno virtual.
2. Instala dependencias:

```bash
pip install -r requirements_dashboard_urbania.txt
```

3. Ejecuta el dashboard:

```bash
streamlit run app_dashboard_urbania.py
```

## Qué incluye

- Resumen de la muestra.
- Distribución de `ln_precio_m2`.
- Ranking de distritos por mediana del precio/m².
- Modelo predictivo con `ln_precio_m2`.
- Simulador para estimar precio por m² y precio total.
- Ejemplos rápidos de predicción.

Las predicciones están en la misma moneda del dataset.
