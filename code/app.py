import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Análise arremessos Kobe.
""")

df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

# st.write(df_prod)
# st.write(df_dev)
fignum = plt.figure(figsize=(6,4))

sns.distplot(
    df_dev.prediction_score_1,
    label='Teste',
    ax = plt.gca()
)

sns.distplot(
    df_prod.predict_score,
    label='Prod',
    ax = plt.gca()
)

plt.title('Monitoramento Desvio de Dados da Saída do modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade de acerto no arremesso')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')

st.pyplot(fignum)

st.write(metrics.classification_report(df_dev.shot_made_flag, df_dev.prediction_label))

# Saída do modelo em dados dev

# ############################################ LEITURA DOS DADOS
# @st.cache_data()
# def load_data(fname):
#     return joblib.load(fname)

# results = load_data(fname)
# model = results[wine_type]['model'] 
# train_data = results[wine_type]['data']
# features = results[wine_type]['features']
# target_col = results[wine_type]['target_col']
# idx_train = train_data.categoria == 'treino'
# idx_test = train_data.categoria == 'teste'
# train_threshold = results[wine_type]['threshold']

# print(f"features {features}")
# print(f"train_data {train_data.columns}")


# ############################################ TITULO
# st.title(f"""
# Sistema Online de Avaliação de Vinhos Tipo {'Tinto' if wine_type == 'red' else 'Branco'}
# """)

# st.markdown(f"""
# Esta interface pode ser utilizada para a explanação dos resultados
# do modelo de classificação da qualidade de vinhos brancos e tintos,
# segundo as variáveis utilizadas para caracterizar os vinhos.

# O modelo selecionado ({wine_type}) foi treinado com uma base total de {idx_train.sum()} e avaliado
# com {idx_test.sum()} novos dados (histórico completo de {train_data.shape[0]} vinhos.

# Os vinhos são caracterizados pelas seguintes variáveis: {features}.
# """)


# ############################################ ENTRADA DE VARIAVEIS
# st.sidebar.header('Entrada de Variáveis')
# form = st.sidebar.form("input_form")
# input_variables = {}

# print(train_data.info())

# for cname in features:
# #     print(f'cname {cname}')
# #     print(train_data[cname].unique())
# #     print(train_data[cname].astype(float).max())
# #     print(float(train_data[cname].astype(float).min()))
# #     print(float(train_data[cname].astype(float).max()))
# #     print(float(train_data[cname].astype(float).mean()))
#     input_variables[cname] = (form.slider(cname.capitalize(),
#                                           min_value = float(train_data[cname].astype(float).min()),
#                                           max_value = float(train_data[cname].astype(float).max()),
#                                           value = float(train_data[cname].astype(float).mean()))
#                                    ) 
                             
# form.form_submit_button("Avaliar")

# ############################################ PREVISAO DO MODELO 
# @st.cache_data
# def predict_user(input_variables):
#     print(f'input_variables {input_variables}')
#     X = pandas.DataFrame.from_dict(input_variables, orient='index').T
#     Yhat = model.predict_proba(X)[0,1]
#     return {
#         'probabilidade': Yhat,
#         'classificacao': int(Yhat >= train_threshold)
#     }

# user_wine = predict_user(input_variables)

# if user_wine['classificacao'] == 0:
#     st.sidebar.markdown("""Classificação:
#     <span style="color:red">*Baixa* qualidade</span>.
#     """, unsafe_allow_html=True)
# else:
#     st.sidebar.markdown("""Classificação:
#     <span style="color:green">*Alta* qualidade</span>.
#     """, unsafe_allow_html=True)

# ############################################ PAINEL COM AS PREVISOES HISTORICAS

# fignum = plt.figure(figsize=(6,4))
# for i in train_data.target.unique():
#     sns.distplot(train_data[train_data[target_col] == i].probabilidade,
#                  label=train_data[train_data[target_col] == i].target_label,
#                  ax = plt.gca())
# # User wine
# plt.plot(user_wine['probabilidade'], 2, '*k', markersize=3, label='Vinho Usuário')

# plt.title('Resposta do Modelo para Vinhos Históricos')
# plt.ylabel('Densidade Estimada')
# plt.xlabel('Probabilidade Vinho Alta Qualidade')
# plt.xlim((0,1))
# plt.grid(True)
# plt.legend(loc='best')
# st.pyplot(fignum)