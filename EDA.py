from networkx import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preparar_dados():
    
    df = pd.read_csv('student_habits_performance.csv')

    print("Dataset")
    print(f"\n-Dimensoes:{df.shape}")
    print(f"\n-Colunas:{df.columns}")
    print(f"\n-valores nulos:{df.isnull().sum().sum()}")
    print(f"\n-duplicados:{df.duplicated().sum()}")
    
    if df.isnull().sum().sum() > 0:
     # Preencher numéricos com mediana
     numeric_cols = df.select_dtypes(include=[np.number]).columns
     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Preencher categóricos com moda
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"\n-verificar valores nulos:{df.isnull().sum().sum()}")
 

    df['Desempenho'] = pd.cut(df['exam_score'],
                              bins=[0, 25, 50, 70, 100],
                              labels=['0-25', '26-50', '51-70', '71-100'],
                              include_lowest=True,
                              right=True)
                      

    distribuicao = df['Desempenho'].value_counts().sort_index()
    print(distribuicao)  



    categorical_columns =['gender', 'diet_quality', 'parental_education_level',
        'internet_quality', 'extracurricular_participation', 'part_time_job']

    for col in categorical_columns:
        if col in df.columns:
            encoded_col=col+'_encoded'
            le = LabelEncoder()
            df[encoded_col] = le.fit_transform(df[col].astype(str))
            print(f"-{encoded_col}")

    encoded_cols = [col for col in df.columns 
                    if col.endswith('_encoded') and col in df.columns]

    print(f"\nColunas codificadas criadas:{len(encoded_cols)}")
    print(encoded_cols)

    final_features = [
        'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating',
    ] + encoded_cols

    print(f"Total de features selecionadas: {len(final_features)}")
    print(f"-Numericas: {len([f for f in final_features if 'encoded' not in f])}")
    print(f"-Categoricas codificadas: {len([f for f in final_features if 'encoded' in f])}")
    print("\nFeatures selecionadas:")
    print(final_features)

    x=df[final_features]
    y=df['Desempenho']

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("DIVISAO DOS DADOS:")
    print(f"-Treino: {X_train.shape[0]} amostras")
    print(f"-Teste: {X_test.shape[0]} amostras")
    print(f"-Features: {X_train.shape[1]}") 

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("NORMALIZACAO:")
    print("Dados normalizados para K-NN")
    print(f"- X_train_scaled shape: {X_train_scaled.shape}")
    print(f"- X_test_scaled shape: {X_test_scaled.shape}")

    faixa_para_intervalo = {
        '0-25': '0-25 pontos',
        '26-50': '26-50 pontos',
        '51-70': '51-70 pontos', 
        '71-100': '71-100 pontos'
    }
    print("\nMAPEAMENTO FAIXA->INTERVALO:")
    for faixa, intervalo in faixa_para_intervalo.items():
        print(f"  {faixa} -> {intervalo}")
    
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler,
        'final_features': final_features,
        'faixa_para_intervalo': faixa_para_intervalo,
        'df': df
    }

if __name__ == "__main__":
    dados = preparar_dados()
    print(" Dados preparados com sucesso!")