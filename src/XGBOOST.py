import xgboost as xgb
import numpy as np

def train_xgboost(X_train, y_train):
    """
    Entrena un modelo XGBoost Regressor para predicción mensual.
    """
    # Aplanamos X_train (batch, seq, feat) -> (batch, seq * feat)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    model.fit(X_train_flat, y_train)
    return model

def predict_xgboost_recursive(model, initial_context, steps, s_test, s_train, input_window, n_feat):
    """
    Predicción recursiva usando XGBoost.
    """
    current_context = initial_context.copy()
    preds = []
    
    for i in range(steps):
        # Preparar input
        inp = current_context[-input_window:].reshape(1, -1)
        p = model.predict(inp)[0]
        preds.append(p)
        
        # Inyectar para la siguiente predicción (usando exógenas futuras de s_test)
        future_idx = len(s_train) + i
        if future_idx < len(s_test):
            next_row = s_test[future_idx].copy()
            next_row[-1] = p # Reemplazar con predicción
            current_context = np.vstack([current_context, next_row])
            
    return np.array(preds)
