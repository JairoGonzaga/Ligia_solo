import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Detector de Fraude",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Tema escuro moderno */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Título principal */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Resultado cards */
    .result-fraud {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255, 65, 108, 0.3);
    }
    
    .result-safe {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(56, 239, 125, 0.3);
    }
    
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    
    .probability-text {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4);
    }
    
    /* Feature importance */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)


# Função para carregar modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('Modelo_infos/modelo_stacking.joblib')
        scaler = joblib.load('Modelo_infos/scaler.joblib')
        power_transformer = joblib.load('Modelo_infos/power_transformer.joblib')
        threshold = joblib.load('Modelo_infos/best_threshold.joblib')
        feature_names = joblib.load('Modelo_infos/feature_names.joblib')
        return model, scaler, power_transformer, threshold, feature_names
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar modelos: {e}")
        st.info("Execute primeiro o notebook model.ipynb para gerar os arquivos do modelo.")
        return None, None, None, None, None


model, scaler, power_transformer, threshold, feature_names = load_models()

st.markdown('<h1 class="main-title">🔍 Detector de Fraude</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema inteligente de detecção de transações fraudulentas usando Machine Learning</p>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

if model is not None:
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        st.markdown("### 📊 Dados da Transação")
        
        # Criar tabs para diferentes métodos de input
        tab1, tab2 = st.tabs(["📝 Entrada Manual", "📁 Upload CSV"])
        
        with tab1:
            # Grid de inputs para as features
            st.markdown("#### Features PCA (V1-V28)")
            
            # Criar colunas para os inputs V1-V28
            v_features = {}
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i-1) % 4]:
                    v_features[f'V{i}'] = st.number_input(
                        f'V{i}', 
                        value=0.0, 
                        format="%.6f",
                        key=f'v{i}'
                    )
            
            st.markdown("#### Outras Features")
            col1, col2 = st.columns(2)
            with col1:
                time_val = st.number_input('Time (segundos)', value=0.0, format="%.2f")
            with col2:
                amount_val = st.number_input('Amount (valor)', value=0.0, format="%.2f")
            
            st.markdown("")
            predict_button = st.button("🔮 Analisar Transação", use_container_width=True)
            
            if predict_button:
                # Preparar dados
                input_data = pd.DataFrame({
                    **v_features,
                    'Time': [time_val],
                    'Amount': [amount_val]
                })
                
                # Reordenar colunas conforme feature_names
                input_data = input_data[feature_names]
                
                # Pré-processamento
                input_data[["Amount", "Time"]] = scaler.transform(input_data[["Amount", "Time"]])
                input_data = power_transformer.transform(input_data)
                
                # Predição
                proba = model.predict_proba(input_data)[:, 1][0]
                is_fraud = proba >= threshold
                
                # Armazenar resultado na sessão
                st.session_state['result'] = {
                    'proba': proba,
                    'is_fraud': is_fraud,
                    'input_data': input_data
                }
        
        with tab2:
            uploaded_file = st.file_uploader("Faça upload de um arquivo CSV", type=['csv'])
            
            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)
                st.dataframe(df_upload.head(), use_container_width=True)
                
                if st.button("🔮 Analisar Todas as Transações", key="batch_predict"):
                    # Verificar se todas as features estão presentes
                    missing_features = [f for f in feature_names if f not in df_upload.columns]
                    
                    if missing_features:
                        st.error(f"Features faltando: {', '.join(missing_features)}")
                    else:
                        # Preparar dados
                        df_process = df_upload[feature_names].copy()
                        df_process[["Amount", "Time"]] = scaler.transform(df_process[["Amount", "Time"]])
                        df_process = power_transformer.transform(df_process)
                        
                        # Predição
                        probas = model.predict_proba(df_process)[:, 1]
                        predictions = (probas >= threshold).astype(int)
                        
                        # Adicionar resultados
                        df_upload['Probabilidade_Fraude'] = probas
                        df_upload['Fraude'] = predictions
                        
                        # Estatísticas
                        n_fraud = predictions.sum()
                        n_total = len(predictions)
                        
                        st.success(f"✅ Análise concluída! {n_fraud} fraudes detectadas em {n_total} transações")
                        
                        # Mostrar resultados
                        st.dataframe(
                            df_upload[['Probabilidade_Fraude', 'Fraude']].style.background_gradient(
                                subset=['Probabilidade_Fraude'], cmap='RdYlGn_r'
                            ),
                            use_container_width=True
                        )
                        
                        # Download dos resultados
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            "📥 Download Resultados",
                            csv,
                            "resultados_fraude.csv",
                            "text/csv",
                            use_container_width=True
                        )
    
    with col_result:
        st.markdown("### 🎯 Resultado da Análise")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            proba = result['proba']
            is_fraud = result['is_fraud']
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de Fraude", 'font': {'size': 18, 'color': 'white'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ff416c" if is_fraud else "#38ef7d"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(56, 239, 125, 0.3)'},
                        {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(255, 65, 108, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Card de resultado
            if is_fraud:
                st.markdown("""
                <div class="result-fraud">
                    <p class="result-text">⚠️ FRAUDE DETECTADA</p>
                    <p class="probability-text">Transação identificada como potencialmente fraudulenta</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-safe">
                    <p class="result-text">✅ TRANSAÇÃO SEGURA</p>
                    <p class="probability-text">Nenhuma atividade suspeita detectada</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Informações adicionais
            st.markdown("")
            st.markdown("#### 📈 Detalhes da Análise")
            
            st.metric("Probabilidade", f"{proba*100:.2f}%")
        else:
            st.info("👆 Insira os dados da transação e clique em 'Analisar Transação'")
            
            # Placeholder visual
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=0,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "gray"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(56, 239, 125, 0.1)'},
                        {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.1)'},
                        {'range': [70, 100], 'color': 'rgba(255, 65, 108, 0.1)'}
                    ]
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Seção de informações do modelo
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Sobre o Modelo"):
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("""
            **🤖 Algoritmos**
            - RandomForest
            - XGBoost
            - LightGBM
            - Meta-learner: Logistic Regression
            """)
        
        with col_info2:
            st.markdown("""
            **⚙️ Pré-processamento**
            - StandardScaler (Time, Amount)
            - PowerTransformer (Yeo-Johnson)
            - BorderlineSMOTE
            - RandomUnderSampler
            """)
        
        with col_info3:
            st.markdown(f"""
            **📊 Configurações**
            - Número de Features: {len(feature_names)}
            - Método: Stacking Ensemble
            """)

else:
    st.warning("⚠️ Modelos não encontrados. Execute o notebook model.ipynb primeiro para gerar os arquivos necessários.")
    
    st.markdown("""
    ### 📋 Arquivos necessários:
    - `modelo_stacking.joblib`
    - `scaler.joblib`
    - `power_transformer.joblib`
    - `best_threshold.joblib`
    - `feature_names.joblib`
    """)

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">Desenvolvido com Streamlit e Scikit-learn | Modelo de Detecção de Fraude</p>',
    unsafe_allow_html=True
)
