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

import pandas as pd
import json
import gzip
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer
)


def solucion():

    def cargar_archivo(ruta):
        return pd.read_csv(ruta, index_col=False, compression="zip")


    def depurar(df):
        df = df.rename(columns={"default payment next month": "default"})
        df = df.drop(columns=["ID"])
        df = df.dropna()
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
        return df


    def construir_pipeline():
        cat_vars = ["SEX", "EDUCATION", "MARRIAGE"]

        codificador = ColumnTransformer(
            transformers=[
                ("categoricas", OneHotEncoder(handle_unknown="ignore"), cat_vars)
            ],
            remainder="passthrough"
        )

        modelo = Pipeline(steps=[
            ("preprocesamiento", codificador),
            ("clasificador", RandomForestClassifier(random_state=42))
        ])

        return modelo


    def ajustar_modelo(pipe, x_train, y_train):
        grid = {
            "clasificador__n_estimators": [100, 200, 300],
            "clasificador__max_depth": [None, 10, 20, 30],
            "clasificador__min_samples_split": [2, 5, 10],
        }

        puntuador = make_scorer(balanced_accuracy_score)

        optimizador = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=puntuador,
            cv=10,
            n_jobs=-1,
            verbose=1
        )

        optimizador.fit(x_train, y_train)
        return optimizador


    def guardar_modelo(modelo, ruta_salida="files/models/model.pkl.gz"):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        with gzip.open(ruta_salida, "wb") as archivo:
            pickle.dump(modelo, archivo)



    def generar_metricas(modelo, x_train, y_train, x_test, y_test, ruta="files/output/metrics.json"):

        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        registros = []

        for nombre, X, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            pred = modelo.predict(X)
            registro = {
                "type": "metrics",
                "dataset": nombre,
                "precision": precision_score(y, pred),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "recall": recall_score(y, pred),
                "f1_score": f1_score(y, pred)
            }
            registros.append(registro)

        with open(ruta, "w", encoding="utf-8") as salida:
            for r in registros:
                salida.write(json.dumps(r) + "\n")

    def agregar_matrices_conf(modelo, x_train, y_train, x_test, y_test, ruta="files/output/metrics.json"):

        with open(ruta, "r", encoding="utf-8") as archivo:
            contenido = [json.loads(linea) for linea in archivo]

        for nombre, X, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            pred = modelo.predict(X)
            matriz = confusion_matrix(y, pred, labels=[0, 1])

            dato_cm = {
                "type": "cm_matrix",
                "dataset": nombre,
                "true_0": {
                    "predicted_0": int(matriz[0][0]),
                    "predicted_1": int(matriz[0][1]),
                },
                "true_1": {
                    "predicted_0": int(matriz[1][0]),
                    "predicted_1": int(matriz[1][1]),
                }
            }

            contenido.append(dato_cm)

        with open(ruta, "w", encoding="utf-8") as archivo:
            for elemento in contenido:
                archivo.write(json.dumps(elemento) + "\n")


    ruta = "files/input/"

    df_train = cargar_archivo(ruta + "train_data.csv.zip")
    df_test = cargar_archivo(ruta + "test_data.csv.zip")

    df_train = depurar(df_train)
    df_test = depurar(df_test)

    x_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]

    x_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]

    pipe = construir_pipeline()
    mejor_modelo = ajustar_modelo(pipe, x_train, y_train)

    guardar_modelo(mejor_modelo)
    generar_metricas(mejor_modelo, x_train, y_train, x_test, y_test)
    agregar_matrices_conf(mejor_modelo, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    solucion()