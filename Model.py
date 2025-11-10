from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_models(x_train,x_trainscaled, y_train):
    # Treinar Random Forest e k-nn
    rf_model=RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(x_train, y_train)
    print("Random Forest treinado")

    knn_model=KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_trainscaled,y_train)
    print("k-nn treinado")

    return rf_model,knn_model

def evaluate_models(rf_model, knn_model, X_test, X_test_scaled, y_test):
    print("\n Avaliaçao dos modelos")

    y_pred_rf=rf_model.predict(X_test)
    y_pred_knn=knn_model.predict(X_test_scaled)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    
    print(f"Random Forest -accuracy : {acc_rf:.3f}")
    print(f"K-NN - accuracy: {acc_knn:.3f}")
    
    return acc_rf, acc_knn

def save_models(rf_model, knn_model, scaler, filepath='../models/'):
    import os
    os.makedirs(filepath, exist_ok=True)
    
    joblib.dump(rf_model, filepath + 'random_forest_model.pkl')
    joblib.dump(knn_model, filepath + 'knn_model.pkl') 
    joblib.dump(scaler, filepath + 'scaler.pkl')
    
    print("Modelos salvos!")

if __name__ == "__main__":
    from EDA import preparar_dados

    data = preparar_dados()
    rf_model, knn_model = train_models(
        data['X_train'], data['X_train_scaled'], data['y_train']
    )

    evaluate_models(
        rf_model, knn_model, 
        data['X_test'], data['X_test_scaled'], data['y_test']
    )
    
    save_models(rf_model, knn_model, data['scaler'])
    