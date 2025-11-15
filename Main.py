# main.py
from EDA import preparar_dados
from Model import train_models, Model_evaluation, save_models
from prediction_system import generate_complete_prediction_report
from unsupervised import unsupervised_
import time

def main():
    print(" SISTEMA AVANCADO DE PREVISAO DE DESEMPENHO")
    print("="*60)
    
    # 1. Preparar dados
    print("\n1. A PREPARAR DADOS...")
    start_time = time.time()
    dados = preparar_dados()
    print(f" Dados preparados em {time.time() - start_time:.2f} segundos")
    
    # 2. Treinar modelos com otimização
    print("\n2. A TREINAR OS MODELOS COM HYPERPARAMETER OPTIMIZATION...")
    start_time = time.time()
    rf_model, knn_model, rf_params, knn_params = train_models(
        dados['X_train'], dados['X_train_scaled'], dados['y_train'], use_optimization=True
    )
    print(f"Modelos treinados em {time.time() - start_time:.2f} segundos")
    
    # 3. Avaliação abrangente
    print("\n3. A AVALIAR MODELOS...")
    eval_results = Model_evaluation(
        rf_model, knn_model, 
        dados['X_test'], dados['X_test_scaled'], dados['y_test']
    )
    
    # 4. Salvar modelos
    print("\n4. A SALVAR MODELOS...")
    save_models(rf_model, knn_model, dados['scaler'])
    
    # 5. Demonstração do sistema
    print("\n5.DEMONSTRACAO DO SISTEMA...")
    
    # EXEMPLO DE ALUNO BOM 
    aluno_exemplo_BOM = {
        'age': 20,
        'study_hours_per_day': 4.0,
        'social_media_hours': 2.5, 
        'netflix_hours': 1.0,
        'attendance_percentage': 85,
        'sleep_hours': 7.0,
        'exercise_frequency': 3,
        'mental_health_rating': 7,
        'gender_encoded': 1,
        'diet_quality_encoded': 1, 
        'parental_education_level_encoded': 2,
        'internet_quality_encoded': 1,
        'extracurricular_participation_encoded': 1,
        'part_time_job_encoded': 0
    }
    # EXEMPLO DE ALUNO MAU
    aluno_exemplo_MAU = {
    'age': 21,
    'study_hours_per_day': 0.5,        
    'social_media_hours': 6.0,         
    'netflix_hours': 4.0,              
    'attendance_percentage': 45,       
    'sleep_hours': 4.5,                
    'exercise_frequency': 0,           
    'mental_health_rating': 3,         
    'gender_encoded': 1,
    'diet_quality_encoded': 0,         
    'parental_education_level_encoded': 0, 
    'internet_quality_encoded': 0,     
    'extracurricular_participation_encoded': 0, 
    'part_time_job_encoded': 1        
    }
    print("\nEXEMPLO DE ALUNO BOM:")
    prediction_BOM = generate_complete_prediction_report(
        aluno_exemplo_BOM, rf_model, knn_model, dados['scaler'], 
        dados['faixa_para_intervalo'], "Aluno Bom"
    )
    print("\nEXEMPLO DE ALUNO MAU:")
    prediction_MAU = generate_complete_prediction_report(
        aluno_exemplo_MAU, rf_model, knn_model, dados['scaler'], 
        dados['faixa_para_intervalo'], "Aluno Mau"
    )
    

    print("=" * 30)
    print("\nUNSUPERVISED K-MEANS")
    unsupervised_()


if __name__ == "__main__":
    main()