import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Entrenamiento con datos escalados
    regressor = RandomForestRegressor(random_state=0, max_depth=2, n_estimators=10)
    regressor.fit(X_train, y_train_encoded)
    y_pred_rr = regressor.predict(X_train)

    # Entrenamiento con datos sin escalar
    regressor_scaled = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
    regressor_scaled.fit(X_train, y_train_encoded)
    y_pred_prep_rr = regressor_scaled.predict(X_train)

    # Calcular errores
    error_scaled = mean_squared_error(y_train_encoded, y_pred_prep_rr)
    error_unscaled = mean_squared_error(y_train_encoded, y_pred_rr)

    # Devuelve los 5 valores esperados
    return regressor, regressor_scaled, error_scaled, error_unscaled, y_train_encoded

def load_or_train_model(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    try:
        # Intentar cargar el modelo desde un archivo
        regressor = joblib.load('random_forest_model.pkl')
        regressor_scaled = joblib.load('random_forest_model_scaled.pkl')
        return regressor, regressor_scaled, None, None, y_train_encoded
    except FileNotFoundError:
        # Si no se encuentra el modelo, entrenar un nuevo modelo
        regressor, regressor_scaled, error_scaled, error_unscaled, y_train_encoded = train_model(X_train, y_train)
        # Guardar los modelos entrenados
        joblib.dump(regressor, 'random_forest_model.pkl')
        joblib.dump(regressor_scaled, 'random_forest_model_scaled.pkl')
        return regressor, regressor_scaled, error_scaled, error_unscaled, y_train_encoded
    
def evaluate_result(y_pred, y, y_prep_pred, y_prep, metric):
    print(metric.__name__, "WITHOUT preparation:", metric(y_pred, y, average='weighted'))
    print(metric.__name__, "WITH preparation:", metric(y_prep_pred, y_prep, average='weighted'))



