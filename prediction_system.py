# src/03_prediction_system.py
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV

def load_trained_models():
    """Carrega os modelos treinados"""
    try:
        rf_model = joblib.load('../models/random_forest_model.pkl')
        knn_model = joblib.load('../models/knn_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        return rf_model, knn_model, scaler
    except:
        print("Modelos não encontrados. Execute o treinamento primeiro.")
        return None, None, None

def predict_student_performance(student_data, rf_model, knn_model, scaler, faixa_para_intervalo):
    """
    Faz previsao para um aluno
    student_data: dict ou DataFrame com as features
    """
    # Converter para DataFrame se for dict
    if isinstance(student_data, dict):
        student_df = pd.DataFrame([student_data])
    else:
        student_df = student_data
    
    # Previsão Random Forest
    faixa_rf = rf_model.predict(student_df)[0]
    
    # Previsão K-NN (precisa de scaling)
    student_scaled = scaler.transform(student_df)
    faixa_knn = knn_model.predict(student_scaled)[0]
    
    # Verificar concordância
    concordancia = faixa_rf == faixa_knn
    confianca = "Alta" if concordancia else "Media"
    
    # Resultado final (usar RF se concordam, ou ambos se discordam)
    if concordancia:
        faixa_final = faixa_rf
    else:
        faixa_final = f"{faixa_rf}/{faixa_knn}"
    
    return {
        'faixa_prevista': faixa_final,
        'intervalo': faixa_para_intervalo.get(faixa_rf, 'N/A'),
        'confianca': confianca,
        'random_forest': faixa_rf,
        'knn': faixa_knn,
        'concordancia': concordancia
    }

def show_individual_predictions(prediction, student_name="Aluno"):
    """Mostra as previsoes individuais de cada modelo"""
    print("="*50)
    print(f"PREVISOES INDIVIDUAIS DOS MODELOS")
    print("="*50)
    print(f"Estudante: {student_name}")
    print(f"Random Forest: {prediction['random_forest']}")
    print(f"K-NN: {prediction['knn']}")
    print(f"Concordancia: {' SIM' if prediction['concordancia'] else ' NAO'}")
    print("="*50)

def generate_report(prediction, student_name="Aluno"):
    """Gera relatorio de previsao final"""
    print("\n" + "="*50)
    print(f"RELATORIO FINAL - PREVISAO COMBINADA")
    print("="*50)
    print(f"Estudante: {student_name}")
    print(f"Desempenho previsto: {prediction['faixa_prevista']}")
    print(f"Intervalo de nota: {prediction['intervalo']}")
    print(f"Nivel de confianca: {prediction['confianca']}")
    print("="*50)
    
    # Recomendações
    recomendacoes = {
        '0-25': "Apoio URGENTE",
        '26-50': "Acompanhamento proximo", 
        '51-70': "Orientaçao de estudo",
        '71-100': "Manter excelencia"
    }
    
    faixa_principal = prediction['random_forest']
    if faixa_principal in recomendacoes:
        print(f"RECOMENDACAO: {recomendacoes[faixa_principal]}")

def generate_complete_prediction_report(student_data, rf_model, knn_model, scaler, faixa_para_intervalo, student_name="Aluno"):
    """Gera um relatório completo mostrando primeiro as previsões individuais e depois a combinação"""
    
    # Fazer previsão
    prediction = predict_student_performance(
        student_data, rf_model, knn_model, scaler, faixa_para_intervalo
    )
    
    # Mostrar previsões individuais
    show_individual_predictions(prediction, student_name)
    
    # Mostrar resultado combinado
    generate_report(prediction, student_name)
    
    return prediction

if __name__ == "__main__":
    rf_model, knn_model, scaler = load_trained_models()
    
    if rf_model is not None:
        # Exemplo de uso
        from EDA import preparar_dados
        data = preparar_dados()
        
        # Testar com primeiro aluno do teste
        sample_student = data['X_test'].iloc[0:1]
        actual_performance = data['y_test'].iloc[0]
        
        prediction = generate_complete_prediction_report(
            sample_student, rf_model, knn_model, scaler, 
            data['faixa_para_intervalo'], "Aluno Exemplo"
        )
        
        print(f"\nDESEMPENHO REAL: {actual_performance}")