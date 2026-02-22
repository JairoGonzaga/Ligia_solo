# 🔍 LIGIA - Detector de Fraude em Transações

Sistema de Machine Learning para detecção de transações fraudulentas utilizando **Stacking Ensemble** com RandomForest, XGBoost e LightGBM.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Pipeline do Modelo](#pipeline-do-modelo)
- [Resultados](#resultados)

## 📖 Sobre o Projeto

O LIGIA é um sistema de detecção de fraudes em transações financeiras baseado em técnicas avançadas de Machine Learning. O projeto utiliza um ensemble de modelos (Stacking) para maximizar a capacidade de identificação de transações fraudulentas, lidando com o problema de desbalanceamento de classes comum neste tipo de aplicação.

### Características Principais

- **Stacking Ensemble** com 3 modelos base (RandomForest, XGBoost, LightGBM)
- **Meta-learner**: Regressão Logística
- **Técnicas de balanceamento**: BorderlineSMOTE + RandomUnderSampler
- **Interface web** interativa com Streamlit
- **Otimização de threshold** para maximizar F2-Score

## 📁 Estrutura do Projeto

```
LIGIA/
├── Dados/
│   ├── train.csv          # Dados de treino originais
│   ├── test.csv           # Dados de teste
│   ├── Xtrain.csv         # Features de treino
│   └── ytrain.csv         # Labels de treino
├── Modelo_infos/
│   ├── modelo_stacking.joblib    # Pipeline do modelo treinado
│   ├── scaler.joblib             # StandardScaler (Time, Amount)
│   ├── power_transformer.joblib  # PowerTransformer (Yeo-Johnson)
│   ├── best_threshold.joblib     # Threshold otimizado
│   └── feature_names.joblib      # Nomes das features
├── EDA.ipynb              # Análise Exploratória dos Dados
├── model.ipynb            # Treinamento do Modelo
├── Inference.py           # Aplicação Streamlit
├── requirements.txt       # Dependências do projeto
├── submission.csv         # Predições para submissão
└── README.md
```

## 🛠️ Tecnologias Utilizadas

| Categoria | Tecnologias |
|-----------|-------------|
| **Linguagem** | Python 3.8+ |
| **ML/DL** | Scikit-learn, XGBoost, LightGBM |
| **Balanceamento** | Imbalanced-learn (SMOTE, Undersampling) |
| **Visualização** | Matplotlib, Seaborn, Plotly |
| **Interface** | Streamlit |
| **Manipulação de Dados** | Pandas, NumPy |

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/LIGIA.git
cd LIGIA
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

### 1. Análise Exploratória (EDA)

Execute o notebook `EDA.ipynb` para visualizar:
- Distribuição das classes (fraude vs não fraude)
- Análise das features V1-V28 (componentes PCA)
- Estatísticas de Time e Amount
- Correlações entre variáveis

### 2. Treinamento do Modelo

Execute o notebook `model.ipynb` para:
- Pré-processar os dados (StandardScaler + PowerTransformer)
- Treinar o modelo Stacking Ensemble
- Otimizar o threshold de decisão
- Salvar os artefatos do modelo

### 3. Interface de Inferência

```bash
streamlit run Inference.py
```

A interface permite:
- **Entrada Manual**: Inserir valores das features V1-V28, Time e Amount
- **Upload CSV**: Analisar múltiplas transações de uma vez
- **Visualização**: Gauge chart com probabilidade de fraude

## 🔧 Pipeline do Modelo

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRÉ-PROCESSAMENTO                        │
├─────────────────────────────────────────────────────────────────┤
│  1. StandardScaler (Time, Amount)                               │
│  2. PowerTransformer (Yeo-Johnson) - todas as features          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BALANCEAMENTO                             │
├─────────────────────────────────────────────────────────────────┤
│  1. BorderlineSMOTE (classe 1 → 1000 amostras)                  │
│  2. RandomUnderSampler (classe 0 → 60000 amostras)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STACKING ENSEMBLE                           │
├─────────────────────────────────────────────────────────────────┤
│  Modelos Base:                                                  │
│  ├── RandomForest (n_estimators=400, max_depth=25)              │
│  ├── XGBoost (n_estimators=400, lr=0.01, max_depth=25)          │
│  └── LightGBM (n_estimators=400, lr=0.01)                       │
│                                                                 │
│  Meta-Learner: LogisticRegression (C=0.1)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OTIMIZAÇÃO DE THRESHOLD                       │
├─────────────────────────────────────────────────────────────────┤
│  Maximização do F2-Score (favorece Recall)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Features

O modelo utiliza 30 features:

| Feature | Descrição |
|---------|-----------|
| **V1 - V28** | Componentes PCA (anonimizadas) |
| **Time** | Segundos desde a primeira transação |
| **Amount** | Valor da transação |

## 📈 Resultados

O modelo é avaliado utilizando métricas apropriadas para dados desbalanceados:

| Métrica | Descrição |
|---------|-----------|
| **ROC-AUC** | Área sob a curva ROC |
| **PR-AUC** | Área sob a curva Precision-Recall |
| **F2-Score** | F-beta com β=2 (prioriza Recall) |
| **Recall** | Capacidade de detectar fraudes |
| **Precision** | Proporção de alertas corretos |

## 📝 Notas

- O threshold de decisão é otimizado para maximizar o **F2-Score**, priorizando a detecção de fraudes (Recall) sobre a precisão
- Os dados de treino são extremamente desbalanceados (~0.17% de fraudes), por isso técnicas de balanceamento são essenciais
- As features V1-V28 são resultado de transformação PCA por motivos de confidencialidade

## 📄 Licença

Este projeto está sob a licença MIT.

---

**Desenvolvido com ❤️ usando Python e Scikit-learn**
