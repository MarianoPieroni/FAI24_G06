# src/association_rules_final.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from unsupervised import analisar_clusters_kmeans

def association_rules_():
    
    # Carregar dados(o (modelo_kmeans) nao esta sendo usado pois so precisamos do df, e a funçao retorna 2 valores por isso a colocamos)
    modelo_kmeans, df_com_clusters = analisar_clusters_kmeans()

    print("\nAlgoritmo Apriori")
    if df_com_clusters is None:
        print("não encontrados")
        return None
    
    print(f"dados: {df_com_clusters.shape[0]} estudantes")
    
    #preparar os dados
    # copia para nao modificar os dados originais
    df_preparado = df_com_clusters.copy()
    df_preparado['estudo_cat'] = pd.cut(df_preparado['study_hours_per_day'], 
                                      bins=[0, 2, 4, 24], 
                                      labels=['baixo', 'medio', 'alto'])
    
    df_preparado['redes_cat'] = pd.cut(df_preparado['social_media_hours'], 
                                     bins=[0, 2, 4, 24], 
                                     labels=['baixo', 'medio', 'alto'])
    
    # prepara os dados de cada aluno para o apriori analizar
    transacoes = []
    for _, estudante in df_preparado.iterrows():
        transacao = [
            f"Estudo_{estudante['estudo_cat']}",
            f"RedesSociais_{estudante['redes_cat']}",
            f"Sono_{estudante['sleep_hours']:.0f}h",
            f"Exercicio_{estudante['exercise_frequency']}",
            f"Trabalho_{estudante['part_time_job']}",
            f"Cluster_{estudante['Cluster'] + 1}"
        ]
        transacoes.append(transacao)
    
    # Converter para binário
    te = TransactionEncoder()
    te_array = te.fit(transacoes).transform(transacoes)
    df_binario = pd.DataFrame(te_array, columns=te.columns_)
    
    print(f"Transações criadas: {len(transacoes)}")
    
    #variáveis que armazena os melhores resultados
    melhor_rules = None
    melhor_combinacao = None
    
    #testa 2 valores de suporte mínimo
    for min_support in [0.05, 0.08]:
        #para cada suporte, testa 2 valores de confiança mínima
        for min_confidence in [0.6, 0.7]:

        # Aplica algoritmo Apriori para encontrar itemsets frequentes
            frequent_itemsets = apriori(df_binario, 
                                      min_support=min_support, #a frequencia minima que um itemset deve ter
                                      use_colnames=True) #usa os nomes das colunas em vez de índices
        # Verifica se encontrou algum itemset frequente    
            if len(frequent_itemsets) > 0:
                #Gera regras de associação a partir dos itemsets frequentes
                rules = association_rules(frequent_itemsets, 
                                        metric="confidence", #usa confiança como métrica para filtrar
                                        min_threshold=min_confidence) #confiança mínima que uma regra deve ter
                
                if len(rules) > 0:
                    regras_boas = rules[rules['lift'] > 1.2]
                    # Atualiza as melhores regras se é a primeira combinação testada ou se encontrou regras melhores
                    if melhor_rules is None or len(regras_boas) > len(melhor_rules):
                        melhor_rules = regras_boas
                        melhor_combinacao = (min_support, min_confidence)
    
    print(f"melhores parametros: support={melhor_combinacao[0]}, confidence={melhor_combinacao[1]}")
    
    # refaz o algoritmo Apriori com melhores parâmetros
    frequent_itemsets = apriori(df_binario, 
                              min_support=melhor_combinacao[0], 
                              use_colnames=True)
    #gera as regras definitivas
    rules = association_rules(frequent_itemsets, 
                            metric="confidence", 
                            min_threshold=melhor_combinacao[1])
    
    #INTERMEDIATE AND FINAL RESULTS
    print(f"\nINTERMEDIATE AND FINAL RESULTS")
    print("-" * 40)
    print(f"Itemsets frequentes: {len(frequent_itemsets)}")
    print(f"Regras geradas: {len(rules)}")
    
    # Filtrar regras relevantes
    regras_relevantes = rules[rules['lift'] > 1.6]
    print(f"Regras relevantes (lift > 1.2): {len(regras_relevantes)}")
    
    # Análise por cluster
    print(f"\nANÁLISE POR CLUSTER:")
    print("=" * 50)
    
    #percorre todos os clusters
    for cluster_num in sorted(df_preparado['Cluster'].unique()):
        print(f"\nCLUSTER {cluster_num + 1}") 
        
        #filtra as regras relevantes XX
        regras_cluster = regras_relevantes[regras_relevantes['consequents'].apply(
                lambda x: f"Cluster_{cluster_num + 1}" in str(x)
            )
        ]
        #verifica se encontrou regras
        if len(regras_cluster) > 0:
            # Percorre cada regra do cluster
            for i, (_, regra) in enumerate(regras_cluster.iterrows()):
                #Para cada item nos antecedentes da regra, mantem apenas os q n começam com cluster
                antecedentes = [str(x) for x in regra['antecedents'] if not x.startswith('Cluster_')]
                #printa se for valido
                if antecedentes:
                    print(f"  {i+1}. SE {antecedentes}")
                    print(f"     Conf: {regra['confidence']:.1%} | Lift: {regra['lift']:.2f}")
        else:
            print("Nenhuma regra boa")
    
    #PERFORMANCE METRICS ANALYSIS
    print(f"\nanalise de performace")
    print("-" * 40)
    
    if len(regras_relevantes) > 0:
        print(f"metricas das {len(regras_relevantes)} regras relevantes:")
        print(f"Suporte médio: {regras_relevantes['support'].mean():.3f}")
        print(f"Confiança média: {regras_relevantes['confidence'].mean():.3f}")
        print(f"Lift médio: {regras_relevantes['lift'].mean():.3f}")
        
        # Uma análise extra essencial
        alta_confianca = len(regras_relevantes[regras_relevantes['confidence'] > 0.7])
        print(f"Regras com alta confiança (>70%): {alta_confianca}")
    
    return regras_relevantes, df_preparado

if __name__ == "__main__":
    rules, dados_preparados = association_rules_()