from flask import Flask, render_template, jsonify
import matplotlib.pyplot as plt
import io
import base64
from model import load_or_train_model
from utils import load_data, scale_data
from sklearn.metrics import mean_squared_error, f1_score
import numpy as np
from io import StringIO
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)
plt.switch_backend('Agg')

global df, X_train, y_train

df, X_train, y_train = load_data('TotalFeatures-ISCXFlowMeter.csv')

@app.route('/showdatasets')
def show():
    # Convertir `head`, `describe`, y `info` a HTML y devolver como JSON
    data_head_html = df.head(10).to_html(classes="table table-striped")
    data_describe_html = df.describe().to_html(classes="table table-bordered")
    
    buffer = StringIO()
    df.info(buf=buffer)
    data_info_html = buffer.getvalue().replace('\n', '<br>')
    
    return jsonify({
        "data_head": data_head_html,
        "data_describe": data_describe_html,
        "data_info": data_info_html
    })

@app.route('/correlations')
def correlations():
    # Transformar la variable de salida y calcular la matriz de correlación
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]
    corr_matrix = X.corr()
    
    # Obtener las correlaciones con `calss`, ordenadas
    corr_class = corr_matrix["calss"].sort_values(ascending=False).to_frame()
    
    corr_html = corr_class.to_html(classes="table table-striped")
    
    return jsonify({"correlations": corr_html})

@app.route('/plot')
def plot():
    # Cargar o entrenar el modelo
    regressor, regressor_scaled, error_scaled, error_unscaled, y_train_encoded = load_or_train_model(X_train, y_train)
    X_train_scaled = scale_data(X_train)

    y_pred_rr = regressor.predict(X_train_scaled)
    y_pred_prep_rr = regressor_scaled.predict(X_train_scaled)

    n_points = 200
    subset_indices = range(n_points)

    plt.figure(figsize=(12, 6))
    plt.plot(y_train_encoded[:n_points], color='blue', linestyle='-', marker='o', label='Valores reales')
    plt.plot(y_pred_prep_rr[:n_points], color='red', linestyle='--', marker='x', label='Predicción Escalada')
    plt.plot(y_pred_rr[:n_points], color='green', linestyle='--', marker='s', label='Predicción Sin Escalar')
    plt.title('Comparación de Predicciones: Escalado vs Sin Escalar')
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return jsonify({"plot_url": plot_url})

@app.route('/evaluate')
def evaluate():
    X_train_scaled = scale_data(X_train)
    clf_rnd_scaled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd_scaled.fit(X_train_scaled, y_train)
    
    clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    
    y_train_pred = clf_rnd.predict(X_train)
    y_train_prep_pred = clf_rnd_scaled.predict(X_train_scaled)
    
    f1_without_prep = f1_score(y_train, y_train_pred, average='weighted')
    f1_with_prep = f1_score(y_train, y_train_prep_pred, average='weighted')
    
    return jsonify({
        "f1_score_without_preparation": f1_without_prep,
        "f1_score_with_preparation": f1_with_prep
    })

if __name__ == '__main__':
    app.run(debug=True)
