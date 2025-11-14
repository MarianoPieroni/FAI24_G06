# src/unsupervised_learning.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  #reponsavel pela avaliaçao da distancia do kmeans

def analisar_clusters_kmeans():
    # Carregar dados
    try:
        df = pd.read_csv('student_habits_performance.csv')
        #print(f"Dataset carregado {df.shape[0]}")
    except:
        print("Arquivo não encontrado")
        return None, None
    
    # Preparando dados para clustering
    features = ['study_hours_per_day', 'social_media_hours', 'sleep_hours', 
                'exercise_frequency', 'attendance_percentage']
    
    dados_clustering = df[features].copy()
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados_clustering) # Normalização

    k_otimo = encontrar_melhor_k(dados_normalizados)

    #mudar aqui para ter mais clusters
 #   kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    #otimizando
    kmeans = KMeans(n_clusters=k_otimo, init='k-means++', random_state=42, n_init=10)
    #fit_perdict = funçao principal do kmeans onde faz os 3 passos 
    clusters = kmeans.fit_predict(dados_normalizados)
    
    df_com_clusters = df.copy()
    df_com_clusters['Cluster'] = clusters
    
    # Estatísticas por cluster
    for cluster in sorted(df_com_clusters['Cluster'].unique()):
        cluster_data = df_com_clusters[df_com_clusters['Cluster'] == cluster]
        
        print(f"\nCLUSTER {cluster + 1} ({len(cluster_data)} estudantes):")
        print(f"Notas: {cluster_data['exam_score'].mean():.1f}")
        print(f"Horas estudo/dia: {cluster_data['study_hours_per_day'].mean():.1f}h")
        print(f"Horas no insta: {cluster_data['social_media_hours'].mean():.1f}h")
        print(f"Horas sono: {cluster_data['sleep_hours'].mean():.1f}h")
        print(f"Presença: {cluster_data['attendance_percentage'].mean():.1f}%")
    
    return kmeans, df_com_clusters

def unsupervised_():
    try:
        modelo_kmeans, dados_clusters = analisar_clusters_kmeans()
        return modelo_kmeans, dados_clusters
    except Exception as e:
        print(f"Erro na análise: {e}")
        return None, None
    


def encontrar_melhor_k(dados_normalizados, max_k=8):

    print("\nHYPERPARAMETER OPTIMIZATION")
    print("=" * 30)
    
    resultados = []
    
    print("Testando valores de K:")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(dados_normalizados)
        
        # Calcular métricas
        wcss = kmeans.inertia_
        silhouette = silhouette_score(dados_normalizados, clusters)
        
        resultados.append({
            'k': k,
            'wcss': wcss,
            'silhouette': silhouette
        })
        
        print(f"K={k}: WCSS = {wcss:8.2f} | Silhouette = {silhouette:.3f}")
    
    # Encontrar melhor K baseado no silhouette
    melhor_resultado = max(resultados, key=lambda x: x['silhouette'])
    melhor_k = melhor_resultado['k']
    
    print(f"\nMELHOR K: {melhor_k}")
    print(f"Silhouette: {melhor_resultado['silhouette']:.3f}")
    
    return melhor_k