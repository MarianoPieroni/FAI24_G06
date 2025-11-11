from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import numpy as np
import pandas as pd

def optimize_hyperparameters(x_train, x_trainscaled, y_train):
    """Otimiza hiperparâmetros para ambos os modelos"""
    print("=== HYPERPARAMETER OPTIMIZATION ===")
    
    # Otimização para Random Forest
    print("\n1. OPTIMIZING RANDOM FOREST...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    rf_search.fit(x_train, y_train)
    print(f"Best Random Forest Parameters: {rf_search.best_params_}")
    print(f"Best Random Forest Score: {rf_search.best_score_:.3f}")
    
    # Otimização para K-NN
    print("\n2. OPTIMIZING K-NN...")
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn_search = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    knn_search.fit(x_trainscaled, y_train)
    print(f"Best K-NN Parameters: {knn_search.best_params_}")
    print(f"Best K-NN Score: {knn_search.best_score_:.3f}")
    
    return rf_search.best_estimator_, knn_search.best_estimator_, rf_search.best_params_, knn_search.best_params_

def train_models(x_train, x_trainscaled, y_train, use_optimization=True):
    """Treina modelos com ou sem otimização de hiperparâmetros"""
    
    if use_optimization:
        # Usar modelos otimizados
        rf_model, knn_model, rf_best_params, knn_best_params = optimize_hyperparameters(
            x_train, x_trainscaled, y_train
        )
        print("Models trained with OPTIMIZED hyperparameters")
        
    else:
        # Usar parâmetros padrão (seu código original)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(x_train, y_train)
        
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_trainscaled, y_train)
        
        rf_best_params = "Default parameters"
        knn_best_params = "Default parameters"
        print("Models trained with DEFAULT hyperparameters")
    
    return rf_model, knn_model, rf_best_params, knn_best_params

def comprehensive_evaluation(rf_model, knn_model, X_test, X_test_scaled, y_test):
    """Avaliação abrangente com múltiplas métricas"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Previsões
    y_pred_rf = rf_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test_scaled)
    
    # Métricas para Random Forest
    print("\nRANDOM FOREST PERFORMANCE:")
    print("-" * 40)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
    rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    
    print(f"Accuracy:    {acc_rf:.3f}")
    print(f"Precision:   {prec_rf:.3f}")
    print(f"Recall:      {rec_rf:.3f}")
    print(f"F1-Score:    {f1_rf:.3f}")
    
    # Métricas para K-NN
    print("\nK-NN PERFORMANCE:")
    print("-" * 40)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    prec_knn = precision_score(y_test, y_pred_knn, average='weighted')
    rec_knn = recall_score(y_test, y_pred_knn, average='weighted')
    f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
    
    print(f"Accuracy:    {acc_knn:.3f}")
    print(f"Precision:   {prec_knn:.3f}")
    print(f"Recall:      {rec_knn:.3f}")
    print(f"F1-Score:    {f1_knn:.3f}")
    
    # Matrizes de Confusão
    print("\nCONFUSION MATRICES:")
    print("\nRandom Forest:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nK-NN:")
    print(confusion_matrix(y_test, y_pred_knn))
    
    # Relatórios de Classificação
    print("\nCLASSIFICATION REPORTS:")
    print("\nRandom Forest:")
    print(classification_report(y_test, y_pred_rf))
    print("\nK-NN:")
    print(classification_report(y_test, y_pred_knn))
    
    return {
        'rf_metrics': {'accuracy': acc_rf, 'precision': prec_rf, 'recall': rec_rf, 'f1': f1_rf},
        'knn_metrics': {'accuracy': acc_knn, 'precision': prec_knn, 'recall': rec_knn, 'f1': f1_knn},
        'y_pred_rf': y_pred_rf,
        'y_pred_knn': y_pred_knn
    }

def compare_model_performance(evaluation_results, rf_best_params, knn_best_params):
    """Comparação detalhada do desempenho dos modelos"""
    print("\n" + "="*60)
    print("MODEL COMPARISON & HYPERPARAMETER ANALYSIS")
    print("="*60)
    
    rf_metrics = evaluation_results['rf_metrics']
    knn_metrics = evaluation_results['knn_metrics']
    
    # Comparação de Performance
    print("\nPERFORMANCE COMPARISON:")
    print(f"{'Metric':<12} {'Random Forest':<15} {'K-NN':<10} {'Winner':<10}")
    print("-" * 50)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        rf_val = rf_metrics[metric]
        knn_val = knn_metrics[metric]
        winner = "RF" if rf_val > knn_val else "K-NN" if knn_val > rf_val else "Tie"
        print(f"{metric:<12} {rf_val:<15.3f} {knn_val:<10.3f} {winner:<10}")
    
    # Análise de Hiperparâmetros
    print("\nOPTIMAL HYPERPARAMETERS:")
    print(f"Random Forest: {rf_best_params}")
    print(f"K-NN: {knn_best_params}")
    
    # Recomendação Final
    print("\nFINAL RECOMMENDATION:")
    if rf_metrics['accuracy'] > knn_metrics['accuracy']:
        print("Random Forest is the recommended model for production")
    else:
        print("K-NN is the recommended model for production")

def save_models(rf_model, knn_model, scaler, filepath='../models/'):
    """Salva os modelos treinados (mantido do código original)"""
    import os
    os.makedirs(filepath, exist_ok=True)
    
    joblib.dump(rf_model, filepath + 'random_forest_model.pkl')
    joblib.dump(knn_model, filepath + 'knn_model.pkl') 
    joblib.dump(scaler, filepath + 'scaler.pkl')
    
    print("Models saved successfully!")

if __name__ == "__main__":
    from EDA import preparar_dados

    data = preparar_dados()
    
    # Treinar com otimização de hiperparâmetros
    rf_model, knn_model, rf_params, knn_params = train_models(
        data['X_train'], data['X_train_scaled'], data['y_train'], use_optimization=True
    )

    # Avaliação abrangente
    eval_results = comprehensive_evaluation(
        rf_model, knn_model, 
        data['X_test'], data['X_test_scaled'], data['y_test']
    )
    
    # Comparação de modelos
    compare_model_performance(eval_results, rf_params, knn_params)
    
    save_models(rf_model, knn_model, data['scaler'])