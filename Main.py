# main.py
from EDA import preparar_dados
from Model import train_models, comprehensive_evaluation, compare_model_performance, save_models
from prediction_system import generate_complete_prediction_report
from unsupervised import unsupervised_
import time

def main():
    print(" SISTEMA AVANÇADO DE PREVISÃO DE DESEMPENHO")
    print("="*60)
    
    # 1. Preparar dados
    print("\n1. PREPARANDO DADOS...")
    start_time = time.time()
    dados = preparar_dados()
    print(f" Dados preparados em {time.time() - start_time:.2f} segundos")
    
    # 2. Treinar modelos com otimização
    print("\n2. TREINANDO MODELOS COM HYPERPARAMETER OPTIMIZATION...")
    start_time = time.time()
    rf_model, knn_model, rf_params, knn_params = train_models(
        dados['X_train'], dados['X_train_scaled'], dados['y_train'], use_optimization=True
    )
    print(f"Modelos treinados em {time.time() - start_time:.2f} segundos")
    
    # 3. Avaliação abrangente
    print("\n3. AVALIAÇÃO COMPREENSIVA DOS MODELOS...")
    eval_results = comprehensive_evaluation(
        rf_model, knn_model, 
        dados['X_test'], dados['X_test_scaled'], dados['y_test']
    )
    
    # 4. Comparação de performance
    compare_model_performance(eval_results, rf_params, knn_params)
    
    # 5. Salvar modelos
    print("\n4. SALVANDO MODELOS...")
    save_models(rf_model, knn_model, dados['scaler'])
    
    # 6. Demonstração do sistema
    print("\n5.DEMONSTRAÇÃO DO SISTEMA...")
    
    # Criar aluno hipotético
    aluno_exemplo = {
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
    
    prediction = generate_complete_prediction_report(
        aluno_exemplo, rf_model, knn_model, dados['scaler'], 
        dados['faixa_para_intervalo'], "Aluno Hipotético"
    )
    
    print("\nSISTEMA AVANÇADO PRONTO PARA USO!")

    print("=" * 30)
    print("\nUNSUPERVISED K-MEANS")
    unsupervised_()


if __name__ == "__main__":
    main()