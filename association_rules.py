# src/association_rules.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def association_rules_analysis():
    # Carregar dados
    try:
        df = pd.read_csv('student_habits_performance.csv')
        #print(f"Dataset carregado {df.shape[0]}")
    except:
        print("Arquivo não encontrado")
        return None, None
    
    df_preparado = df.copy()
    
    # Discretizar notas
    df_preparado['nota_categoria'] = pd.cut(df_preparado['exam_score'], 
                                          bins=[0, 60, 75, 90, 100], 
                                          labels=['baixa', 'media', 'boa', 'alto'])
    
    # Discretizar horas de estudo
    df_preparado['estudo_categoria'] = pd.cut(df_preparado['study_hours_per_day'], 
                                            bins=[0, 2, 4, 6, 24], 
                                            labels=['muito_baixo', 'baixo', 'aoderado', 'alto'])
    
    # Discretizar horas de redes sociais
    df_preparado['redes_sociais_categoria'] = pd.cut(df_preparado['social_media_hours'], 
                                                   bins=[0, 1, 3, 5, 24], 
                                                   labels=['baixo', 'moderado', 'alto', 'muito_Alto'])
    
    print("Criando transações...")
    transacoes = []
    
    for _, estudante in df_preparado.iterrows():
        transacao = [
            f"nota_{estudante['nota_categoria']}",
            f"estudo_{estudante['estudo_categoria']}",
            f"redes_sociais_{estudante['redes_sociais_categoria']}",
            f"genero_{estudante['gender']}",
            f"trabalho_{estudante['part_time_job']}",
            f"dieta_{estudante['diet_quality']}",
            f"exercicio_{estudante['exercise_frequency']}",
            f"sono_{estudante['sleep_hours']:.0f}h"
        ]
        transacoes.append(transacao)
    