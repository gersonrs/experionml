import pandas as pd
import streamlit as st

from experionml import ExperionMLClassifier


# Expande a aplicação web por toda a largura da tela
st.set_page_config(layout="wide")

st.sidebar.title("Pipeline")

# Opções de limpeza de dados
st.sidebar.subheader("Limpeza de dados")
scale = st.sidebar.checkbox("Escalonar", value=False, key="scale")
encode = st.sidebar.checkbox("Codificar", value=False, key="encode")
impute = st.sidebar.checkbox("Imputar", value=False, key="impute")

# Opções de modelos
st.sidebar.subheader("Modelos")
models = {
    "gnb": st.sidebar.checkbox("Naive Bayes Gaussiano", value=True, key="gnb"),
    "rf": st.sidebar.checkbox("Floresta Aleatória", value=True, key="rf"),
    "et": st.sidebar.checkbox("Árvores Extras", value=False, key="et"),
    "xgb": st.sidebar.checkbox("XGBoost", value=False, key="xgb"),
    "lgb": st.sidebar.checkbox("LightGBM", value=False, key="lgb"),
}


st.header("Dados")
data = st.file_uploader("Enviar dados:", type="csv")

# Se um conjunto de dados for enviado, mostra uma prévia
if data is not None:
    data = pd.read_csv(data)
    st.text("Prévia dos dados:")
    st.dataframe(data.head())


st.header("Resultados")

if st.sidebar.button("Executar"):
    placeholder = st.empty()  # Espaço vazio para sobrescrever as mensagens
    placeholder.write("Inicializando o experionml...")

    # Inicializa o experionml
    experionml = ExperionMLClassifier(data, verbose=2, random_state=1)

    if scale:
        placeholder.write("Escalonando os dados...")
        experionml.scale()
    if encode:
        placeholder.write("Codificando as variáveis categóricas...")
        experionml.encode(strategy="Target", max_onehot=10)
    if impute:
        placeholder.write("Imputando os valores ausentes...")
        experionml.impute(strat_num="drop", strat_cat="most_frequent")

    placeholder.write("Treinando os modelos...")
    to_run = [key for key, value in models.items() if value]
    experionml.run(models=to_run, metric="f1")

    # Exibe os resultados das métricas
    placeholder.write(experionml.evaluate())

    # Desenha os gráficos
    col1, col2 = st.beta_columns(2)
    col1.write(experionml.plot_roc(title="Curva ROC", display=None))
    col2.write(experionml.plot_prc(title="Curva PR", display=None))

else:
    st.write("Ainda não há resultados. Clique no botão executar.")
