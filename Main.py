# main.py
from EDA import preparar_dados

def main():
    print("SISTEMA DE PREVISÃO DE DESEMPENHO")
    print("="*50)
    
    # 1. Preparar dados
    print("\n1.PREPARANDO DADOS...")
    dados = preparar_dados()
    
if __name__ == "__main__":
    main()