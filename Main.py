# main.py
from EDA import preparar_dados
from Model import train_models, evaluate_models, save_models
from prediction_system import predict_student_performance, generate_report, load_trained_models
def main():
    print("SISTEMA DE PREVISAO DE DESEMPENHO")
    print("="*50)
    
    # 1. Preparar dados
    print("\n1.PREPARANDO DADOS...")
    dados = preparar_dados()
       # 2. Treinar modelos
    print("\n2.TREINANDO MODELOS...")
    rf_model, knn_model = train_models(
        dados['X_train'], dados['X_train_scaled'], dados['y_train']
    )
    
    # 3. Avaliar
    print("\n3.AVALIANDO MODELOS...")
    evaluate_models(
        rf_model, knn_model, 
        dados['X_test'], dados['X_test_scaled'], dados['y_test']
    )
    
    # 4. Salvar modelos
    print("\n4.SALVANDO MODELOS...")
    save_models(rf_model, knn_model, dados['scaler'])

    # 5. Demonstração do sistema
    print("\n5.DEMONSTRACAO DO SISTEMA...")
    
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
    aluno_mau_desempenho = {
    'age': 19,
    'study_hours_per_day': 1.0,
    'social_media_hours': 6.5,
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

    # Usar a nova função que mostra primeiro as previsões individuais
    from prediction_system import generate_complete_prediction_report
    prediction = generate_complete_prediction_report(
        aluno_mau_desempenho, rf_model, knn_model, dados['scaler'], 
        dados['faixa_para_intervalo'], "Aluno Hipotetico"
    )
    
    
if __name__ == "__main__":
    main()