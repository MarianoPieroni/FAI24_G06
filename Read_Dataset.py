import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('student_habits_performance.csv')

print("=== INFORMACoES DO DATASET ===")
print(f"Dimensoes: {df.shape}")
print("\nPrimeiras 5 linhas:")
print(df.head())

print("\ninformacoes do dataset:")
print(df.info())

print("\nTipos de dados:")
print(df.dtypes)

print("\nEstatisticas descritivas:")
print(df.describe())

print("\nvalores nulos por coluna")
print(df.isnull().sum())