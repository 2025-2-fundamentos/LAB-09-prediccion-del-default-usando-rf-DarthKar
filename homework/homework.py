# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501

import pandas as pd
import json
import pickle
import gzip
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)


def depurar_dataset(df):
    datos = df.copy()
    datos = datos.rename(columns={"default payment next month": "default"})
    datos = datos.drop(columns=["ID"])
    datos = datos[(datos["MARRIAGE"] != 0) & (datos["EDUCATION"] != 0)]
    datos.loc[datos["EDUCATION"] > 4, "EDUCATION"] = 4
    datos = datos.dropna()
    return datos


def construir_modelo():
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    codificador = OneHotEncoder(handle_unknown="ignore")

    transformador = ColumnTransformer(
        transformers=[("categoricas", codificador, columnas_categoricas)],
        remainder="passthrough"
    )
    
    bosque = RandomForestClassifier(random_state=42)
    
    flujo = Pipeline([
        ("preparacion", transformador),
        ("modelo", bosque)
    ])
    
    hiperparametros = {
        "modelo__n_estimators": [100, 200, 500],
        "modelo__max_depth": [None, 5, 10],
        "modelo__min_samples_split": [2, 5],
        "modelo__min_samples_leaf": [1, 2]
    }
    
    optimizador = GridSearchCV(
        flujo,
        hiperparametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True
    )
    
    return optimizador


def obtener_metricas(nombre_conjunto, reales, predichos):
    return {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": precision_score(reales, predichos, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(reales, predichos),
        "recall": recall_score(reales, predichos, zero_division=0),
        "f1_score": f1_score(reales, predichos, zero_division=0)
    }


def obtener_matriz_confusion(nombre_conjunto, reales, predichos):
    matriz = confusion_matrix(reales, predichos)
    return {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {"predicted_0": int(matriz[0][0]), "predicted_1": int(matriz[0][1])},
        "true_1": {"predicted_0": int(matriz[1][0]), "predicted_1": int(matriz[1][1])}
    }


if __name__ == "__main__":
    
    datos_entrenamiento = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    datos_prueba = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    
    datos_entrenamiento = depurar_dataset(datos_entrenamiento)
    datos_prueba = depurar_dataset(datos_prueba)
    
    x_entrenamiento = datos_entrenamiento.drop(columns=["default"])
    y_entrenamiento = datos_entrenamiento["default"]
    x_prueba = datos_prueba.drop(columns=["default"])
    y_prueba = datos_prueba["default"]
    
    modelo_final = construir_modelo()
    modelo_final.fit(x_entrenamiento, y_entrenamiento)
    
    Path("files/models").mkdir(parents=True, exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo_final, f)
    
    pred_train = modelo_final.predict(x_entrenamiento)
    pred_test = modelo_final.predict(x_prueba)
    
    metricas_train = obtener_metricas("train", y_entrenamiento, pred_train)
    metricas_test = obtener_metricas("test", y_prueba, pred_test)
    matriz_train = obtener_matriz_confusion("train", y_entrenamiento, pred_train)
    matriz_test = obtener_matriz_confusion("test", y_prueba, pred_test)
    
    Path("files/output").mkdir(parents=True, exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        f.write(json.dumps(metricas_train) + "\n")
        f.write(json.dumps(metricas_test) + "\n")
        f.write(json.dumps(matriz_train) + "\n")
        f.write(json.dumps(matriz_test) + "\n")
