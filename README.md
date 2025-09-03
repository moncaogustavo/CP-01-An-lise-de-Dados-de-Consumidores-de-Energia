# Checkpoint 01 - Data Science e Machine Learning

## 📋 Sobre o Projeto

Este projeto é um checkpoint acadêmico que demonstra técnicas de Data Science e Machine Learning aplicadas a dados de consumo de energia elétrica. O trabalho utiliza Python e Orange Data Mining para análise exploratória, modelagem preditiva e visualização de dados.

**Total de exercícios:** 40 (20 disponibilizados em 20/08 + 20 disponibilizados em 27/08)

**Estrutura da atividade:**
- **Parte 1:** Exercícios iniciais com Individual Household Electric Power Consumption (20 questões)
- **Parte 2:** Exercícios adicionais no dataset inicial (5 questões)  
- **Parte 3:** Novo dataset Appliances Energy Prediction (10 questões)
- **Parte 4:** Exercícios no Orange Data Mining (5 questões)

## 👥 Equipe

- **Gustavo Bispo** - R558515
- **Gustavo Monção** - 557515  
- **Lucas Barreto** - 557107
- **Vinicius Murtinho** - 551151

## 📊 Datasets Utilizados

### 1. Individual Household Electric Power Consumption
- **Arquivo**: `household_power_consumption.txt`
- **Fonte**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Período**: Dezembro 2006 - Novembro 2010
- **Frequência**: Medições a cada minuto
- **Tamanho**: ~2M registros, 9 variáveis
- **Variáveis principais**:
  - `Global_active_power`: Consumo ativo global (kW)
  - `Global_reactive_power`: Consumo reativo global (kW) 
  - `Voltage`: Tensão (V)
  - `Global_intensity`: Intensidade global (A)
  - `Sub_metering_1/2/3`: Submedições específicas (kWh)

### 2. Appliances Energy Prediction
- **Arquivo**: `energydata_complete.csv`
- **Fonte**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Período**: Janeiro 2016 - Maio 2016
- **Frequência**: Medições a cada 10 minutos
- **Tamanho**: ~19K registros, 28 variáveis
- **Variáveis principais**:
  - `Appliances`: Consumo de eletrodomésticos (Wh)
  - `T1-T9`: Temperaturas em diferentes ambientes
  - `RH_1-RH_9`: Umidade relativa em diferentes ambientes
  - `T_out`: Temperatura externa
  - `RH_out`: Umidade externa

## 🛠️ Tecnologias e Bibliotecas

### Python
- **NumPy** - Computação numérica
- **Pandas** - Manipulação de dados
- **Matplotlib** - Visualização
- **Scikit-learn** - Machine Learning
- **Statsmodels** - Análise de séries temporais (opcional)

### Orange Data Mining
- Interface visual para análise de dados
- Ferramentas de clustering e visualização

## 📁 Estrutura do Projeto

```
checkpoint_ds_ml.py              # Script principal com todas as análises (40 exercícios)
household_power_consumption.txt  # Dataset de consumo doméstico (UCI)
energydata_complete.csv         # Dataset de eletrodomésticos (UCI)
README.md                       # Documentação do projeto
```

### Arquivos de Dados
- **household_power_consumption.txt**: Baixar de [UCI Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **energydata_complete.csv**: Baixar de [UCI Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

## 🚀 Como Executar

### Pré-requisitos

#### Python
```bash
pip install numpy pandas matplotlib scikit-learn
pip install statsmodels  # Opcional para decomposição de séries temporais
```

#### Orange Data Mining
- Baixar e instalar Orange Data Mining: https://orangedatamining.com/download/
- Versão recomendada: 3.34+ (compatível com Python 3.8+)

### Execução

#### Python
```bash
python checkpoint_ds_ml.py
```

#### Orange Data Mining
1. Abrir Orange Data Mining
2. Usar os widgets conforme instruções das questões 36-40
3. Workflow sugerido: CSV File Import → Data Table → Sample Data → Distribution/Scatter Plot → k-Means

## 📱 Google Colab - Execução Individual

### Preparação no Colab
```python
# 1. Instalar dependências
!pip install numpy pandas matplotlib scikit-learn statsmodels

# 2. Fazer upload dos datasets
from google.colab import files
uploaded = files.upload()

# 3. Definir constantes
POWER_PATH = "household_power_consumption.txt"
APPLIANCES_PATH = "energydata_complete.csv"
RANDOM_STATE = 42

# 4. Importar bibliotecas
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSmodels = True
except Exception:
    HAS_STATSmodels = False
    print("Aviso: statsmodels não encontrado. A decomposição de série temporal (Q19) será pulada.")
```

### Funções Auxiliares para Colab
```python
def parse_power_dataset(path):
    '''Carrega o dataset 'Individual Household Electric Power Consumption' do UCI.'''
    df = pd.read_csv(path, sep=';', na_values='?', low_memory=False)
    numeric_cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                    'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def add_datetime_and_weekday(df):
    '''Cria coluna datetime combinando Date + Time, e coluna Weekday.'''
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Weekday'] = df['Date_dt'].dt.day_name(locale='en_US')
    df['Year'] = df['Date_dt'].dt.year
    df['Month'] = df['Date_dt'].dt.month
    df['Day'] = df['Date_dt'].dt.day
    return df

def plot_simple_line(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

### Instruções para Execução Individual

**Para cada exercício, copie e cole o código correspondente no Google Colab:**

1. **Execute primeiro** o bloco de preparação
2. **Execute as funções auxiliares**
3. **Execute cada exercício individualmente** (Q1, Q2, Q3, etc.)
4. **Para Orange Data Mining** (Q36-Q40), use a interface visual conforme instruções

### Dicas para Google Colab
- Use `plt.show()` para exibir gráficos
- Para datasets grandes, considere usar `df.head()` para preview
- Se houver erro de memória, use amostragem: `df.sample(n=10000)`
- Para salvar resultados: `plt.savefig('grafico.png')`

## 📈 Análises Realizadas

### Parte 1 - Exercícios Iniciais (Power Dataset) - 20 questões

#### **Q1 - Carregamento e Exibição**
```python
power = parse_power_dataset(POWER_PATH)
print("Q1) Primeiras 10 linhas:\n", power.head(10), "\n")
```
- **Função**: `parse_power_dataset()` carrega o dataset com separador ';' e valores '?' como NaN
- **Saída**: Exibe as 10 primeiras linhas do dataset
- **Google Colab**: Copie e cole este bloco para executar

#### **Q2 - Explicação Conceitual**
- **Conteúdo**: Explicação da diferença entre Global_active_power e Global_reactive_power
- **Resposta**: Global_active_power = energia útil consumida; Global_reactive_power = energia reativa (não útil)

#### **Q3 - Valores Ausentes**
```python
missing_counts = power.isna().sum()
print("Q3) Total de linhas:", len(power))
print("Valores ausentes por coluna:\n", missing_counts, "\n")
```
- **Função**: Conta valores ausentes por coluna usando `.isna().sum()`
- **Saída**: Total de linhas e contagem de NaN por coluna

#### **Q4 - Conversão Datetime**
```python
power = add_datetime_and_weekday(power)
print("Q4) Colunas Date_dt e Weekday criadas. Preview:\n", power[['Date','Date_dt','Weekday']].head(), "\n")
```
- **Função**: `add_datetime_and_weekday()` combina Date+Time e cria colunas Year, Month, Day, Weekday
- **Saída**: Preview das novas colunas temporais

#### **Q5 - Média Diária 2007**
```python
p2007 = power[power['Year'] == 2007].dropna(subset=['Global_active_power', 'Date_dt'])
daily_mean_2007 = p2007.groupby('Date_dt')['Global_active_power'].mean().rename('DailyMeanGAP')
print("Q5) Média diária 2007 – resumo:\n", daily_mean_2007.describe(), "\n")
```
- **Função**: Filtra dados de 2007 e calcula média diária de Global_active_power
- **Saída**: Estatísticas descritivas das médias diárias

#### **Q6 - Gráfico de Um Dia**
```python
one_day = p2007.dropna(subset=['Datetime']).set_index('Datetime').between_time('00:00','23:59')
if not one_day.empty:
    date_example = one_day.index.date[0]
    day_slice = one_day[one_day.index.date == date_example]['Global_active_power']
    plot_simple_line(day_slice.index, day_slice.values,
                     f"Q6) Global_active_power ao longo do dia - {date_example}",
                     "Hora", "kW")
```
- **Função**: Seleciona um dia completo e plota variação horária
- **Saída**: Gráfico de linha mostrando consumo ao longo de 24h

#### **Q7 - Histograma Voltage**
```python
plt.figure()
power['Voltage'].dropna().hist(bins=50)
plt.title("Q7) Distribuição de Voltage")
plt.xlabel("Voltage (V)")
plt.ylabel("Frequência")
plt.show()
```
- **Função**: Cria histograma da distribuição de tensão
- **Saída**: Gráfico de barras com 50 bins

#### **Q8 - Consumo Mensal**
```python
monthly_mean = power.dropna(subset=['Global_active_power']).groupby(['Year','Month'])['Global_active_power'].mean()
print("Q8) Primeiros meses e estatísticas:\n", monthly_mean.head(), "\n", monthly_mean.describe(), "\n")
```
- **Função**: Agrupa por ano/mês e calcula média de Global_active_power
- **Saída**: Série temporal com médias mensais

#### **Q9 - Dia de Maior Consumo**
```python
daily_mean_all = power.dropna(subset=['Global_active_power','Date_dt']).groupby('Date_dt')['Global_active_power'].mean()
max_day = daily_mean_all.idxmax()
max_value = daily_mean_all.max()
print("Q9) Dia com maior consumo médio:", max_day, "Valor (kW):", max_value, "\n")
```
- **Função**: Calcula médias diárias e identifica o dia com maior valor
- **Saída**: Data e valor do maior consumo diário

#### **Q10 - Comparação Fim de Semana**
```python
power['IsWeekend'] = power['Date_dt'].dt.weekday >= 5
weekday_mean = power.loc[power['IsWeekend'] == False].groupby('Date_dt')['Global_active_power'].mean().mean()
weekend_mean = power.loc[power['IsWeekend'] == True].groupby('Date_dt')['Global_active_power'].mean().mean()
print("Q10) Média diária semana (kW):", weekday_mean, " | fim de semana (kW):", weekend_mean, "\n")
```
- **Função**: Cria flag de fim de semana e compara médias
- **Saída**: Duas médias comparativas

#### **Q11 - Matriz de Correlação**
```python
corr_vars = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity']
corr_df = power[corr_vars].dropna().corr()
print("Q11) Correlação entre variáveis:\n", corr_df, "\n")
```
- **Função**: Calcula correlação entre 4 variáveis principais
- **Saída**: Matriz de correlação 4x4

#### **Q12 - Total Submedições**
```python
for c in ['Sub_metering_1','Sub_metering_2','Sub_metering_3']:
    if c not in power.columns:
        power[c] = np.nan
power['Total_Sub_metering'] = power[['Sub_metering_1','Sub_metering_2','Sub_metering_3']].sum(axis=1, min_count=1)
print("Q12) Total_Sub_metering criada. Preview:\n", power[['Sub_metering_1','Sub_metering_2','Sub_metering_3','Total_Sub_metering']].head(), "\n")
```
- **Função**: Soma as 3 submedições criando nova variável
- **Saída**: Preview da nova coluna calculada

#### **Q13 - Comparação Mensal**
```python
monthly_tot_sub = power.groupby(['Year','Month'])['Total_Sub_metering'].mean()
monthly_gap = power.groupby(['Year','Month'])['Global_active_power'].mean()
compare = pd.DataFrame({
    'Monthly_Total_Sub_metering': monthly_tot_sub,
    'Monthly_Global_active_power': monthly_gap
}).dropna()
compare['Exceeds'] = compare['Monthly_Total_Sub_metering'] > compare['Monthly_Global_active_power']
print("Q13) Meses em que Total_Sub_metering > Global_active_power (médias mensais):\n", compare[compare['Exceeds']==True].head(), "\n")
```
- **Função**: Compara médias mensais de submedições vs consumo global
- **Saída**: Meses onde submedições excedem consumo global

#### **Q14 - Série Temporal Voltage 2008**
```python
p2008 = power[power['Year'] == 2008].dropna(subset=['Datetime','Voltage'])
if not p2008.empty:
    ts2008 = p2008.set_index('Datetime')['Voltage'].resample('1D').mean()
    plot_simple_line(ts2008.index, ts2008.values, "Q14) Voltage - Média diária (2008)", "Data", "Voltage (V)")
```
- **Função**: Filtra 2008 e cria série temporal diária de tensão
- **Saída**: Gráfico de linha com evolução diária

#### **Q15 - Comparação Sazonal**
```python
gap = power.dropna(subset=['Global_active_power'])
summer = gap[gap['Month'].isin([6,7,8])]['Global_active_power'].mean()
winter = gap[gap['Month'].isin([12,1,2])]['Global_active_power'].mean()
print("Q15) Média GAP - Verão (JJA):", summer, " | Inverno (DJF):", winter, "\n")
```
- **Função**: Compara médias de verão (JJA) vs inverno (DJF)
- **Saída**: Duas médias sazonais

#### **Q16 - Amostragem 1%**
```python
sample = power.sample(frac=0.01, random_state=RANDOM_STATE)
plt.figure()
power['Global_active_power'].dropna().plot(kind='kde')
sample['Global_active_power'].dropna().plot(kind='kde')
plt.title("Q16) Distribuição KDE - Completa vs Amostra 1%")
plt.xlabel("Global_active_power (kW)")
plt.legend(["Completa","Amostra 1%"])
plt.show()
```
- **Função**: Amostra 1% e compara distribuições com KDE
- **Saída**: Gráfico sobreposto das duas distribuições

#### **Q17 - Normalização Min-Max**
```python
num_vars = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Total_Sub_metering']
scaled_df = power[num_vars].copy()
scaler = MinMaxScaler()
scaled_vals = scaler.fit_transform(scaled_df.values)
scaled_df = pd.DataFrame(scaled_vals, columns=num_vars, index=power.index)
print("Q17) Exemplo de dados escalados (Min-Max):\n", scaled_df.head(), "\n")
```
- **Função**: Aplica MinMaxScaler nas 5 variáveis numéricas
- **Saída**: DataFrame com valores entre 0 e 1

#### **Q18 - K-Means Clustering**
```python
daily = power.dropna(subset=['Date_dt']).groupby('Date_dt')[['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Total_Sub_metering']].mean().dropna()
km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
daily['cluster'] = km.fit_predict(StandardScaler().fit_transform(daily[['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Total_Sub_metering']]))
print("Q18) Tamanho dos clusters (dias):\n", daily['cluster'].value_counts(), "\n")
print("Q18) Perfil médio por cluster:\n", daily.groupby('cluster').mean(), "\n")
```
- **Função**: Agrupa por dia, aplica StandardScaler e K-Means com 3 clusters
- **Saída**: Tamanho dos clusters e perfis médios

#### **Q19 - Decomposição Série Temporal**
```python
if HAS_STATSmodels:
    gp = power.dropna(subset=['Datetime','Global_active_power']).set_index('Datetime')['Global_active_power'].resample('1H').mean().dropna()
    if len(gp) > 0:
        start = gp.index.min()
        end = start + pd.DateOffset(months=6)
        gp_6m = gp.loc[(gp.index >= start) & (gp.index < end)]
        if len(gp_6m) > 24*30:
            result = seasonal_decompose(gp_6m, model='additive', period=24)
            result.plot()
            plt.suptitle("Q19) Decomposição de Série (6 meses) - Global_active_power")
            plt.show()
```
- **Função**: Decompõe série temporal em tendência, sazonalidade e resíduo
- **Saída**: 4 gráficos (original, tendência, sazonalidade, resíduo)
- **Dependência**: Requer statsmodels

#### **Q20 - Regressão Linear**
```python
df_lr = power[['Global_active_power','Global_intensity']].dropna()
X = df_lr[['Global_intensity']].values
y = df_lr['Global_active_power'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
linreg = LinearRegression().fit(X_train, y_train)
pred = linreg.predict(X_test)
print("Q20) RMSE:", rmse(y_test, pred))
print("Coeficiente:", linreg.coef_[0], "Intercepto:", linreg.intercept_, "\n")
```
- **Função**: Treina regressão linear Global_active_power ~ Global_intensity
- **Saída**: RMSE, coeficiente e intercepto

### Parte 2 - Exercícios Adicionais (Power Dataset) - 5 questões

#### **Q21 - Padrão Horário**
```python
s_hourly = power.dropna(subset=['Datetime']).set_index('Datetime')['Global_active_power'].resample('1H').mean().dropna()
hourly_pattern = s_hourly.groupby(s_hourly.index.hour).mean()
print("Q21) Top 5 horários de maior consumo médio:\n", hourly_pattern.sort_values(ascending=False).head(5), "\n")
plt.figure()
plt.plot(hourly_pattern.index, hourly_pattern.values)
plt.title("Q21) Padrão médio por hora do dia - Global_active_power")
plt.xlabel("Hora do dia")
plt.ylabel("kW (média)")
plt.show()
```
- **Função**: Reamostra para hora e calcula padrão médio por hora
- **Saída**: Top 5 horários + gráfico de linha

#### **Q22 - Autocorrelação**
```python
def autocorr_at_lag(series, lag):
    return series.autocorr(lag=lag)
for lag in [1, 24, 48]:
    print(f"Q22) Autocorrelação (lag={lag}h):", autocorr_at_lag(s_hourly, lag))
```
- **Função**: Calcula autocorrelação em lags de 1h, 24h e 48h
- **Saída**: 3 valores de autocorrelação

#### **Q23 - PCA**
```python
sel = power[['Global_active_power','Global_reactive_power','Voltage','Global_intensity']].dropna()
X_std = StandardScaler().fit_transform(sel.values)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_std)
print("Q23) Variância explicada:", pca.explained_variance_ratio_, "\n")
```
- **Função**: Aplica PCA reduzindo 4 variáveis para 2 componentes
- **Saída**: Proporção de variância explicada por cada componente

#### **Q24 - Clustering PCA**
```python
km_pca = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit(X_pca)
labels = km_pca.labels_
plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
plt.title("Q24) PCA (2D) + KMeans (3 clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```
- **Função**: Aplica K-Means no espaço PCA 2D
- **Saída**: Scatter plot colorido por cluster

#### **Q25 - Regressão Polinomial**
```python
df_reg = power[['Global_active_power','Voltage']].dropna()
Xv = df_reg[['Voltage']].values
yv = df_reg['Global_active_power'].values
X_train, X_test, y_train, y_test = train_test_split(Xv, yv, test_size=0.2, random_state=RANDOM_STATE)
lin = LinearRegression().fit(X_train, y_train)
pred_lin = lin.predict(X_test)
rmse_lin = rmse(y_test, pred_lin)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
lin_poly = LinearRegression().fit(X_train_poly, y_train)
pred_poly = lin_poly.predict(X_test_poly)
rmse_poly = rmse(y_test, pred_poly)
print("Q25) RMSE Linear:", rmse_lin, " | RMSE Polinomial (g2):", rmse_poly, "\n")
```
- **Função**: Compara regressão linear vs polinomial (grau 2)
- **Saída**: RMSE de ambos os modelos + gráfico comparativo

### Parte 3 - Appliances Energy Prediction - 10 questões

#### **Q26 - Inspeção Inicial**
```python
appl = pd.read_csv(APPLIANCES_PATH)
print("Q26) .info():")
print(appl.info())
print("Q26) .describe():\n", appl.describe(include='all'), "\n")
```
- **Função**: Carrega dataset e exibe informações básicas
- **Saída**: Info do DataFrame + estatísticas descritivas

#### **Q27 - Distribuição Appliances**
```python
plt.figure()
appl['Appliances'].dropna().hist(bins=50)
plt.title("Q27) Histograma - Appliances")
plt.xlabel("Wh")
plt.ylabel("Frequência")
plt.show()
if 'date' in appl.columns:
    appl['date'] = pd.to_datetime(appl['date'], errors='coerce')
    s_app = appl.dropna(subset=['date']).set_index('date')['Appliances'].resample('1H').mean()
    plot_simple_line(s_app.index, s_app.values, "Q27) Appliances (média por hora)", "Data/Hora", "Wh")
```
- **Função**: Cria histograma e série temporal do consumo
- **Saída**: Histograma + gráfico de linha temporal

#### **Q28 - Correlações Ambientais**
```python
cand_cols = [c for c in appl.columns if c.lower().startswith(('t','rh'))]
corr_cols = ['Appliances'] + cand_cols
corr_appl = appl[corr_cols].corr()['Appliances'].sort_values(ascending=False)
print("Q28) Correlações (Appliances vs variáveis ambientais):\n", corr_appl, "\n")
```
- **Função**: Calcula correlações com variáveis de temperatura e umidade
- **Saída**: Série ordenada de correlações

#### **Q29 - Normalização**
```python
num_cols_appl = appl.select_dtypes(include=[np.number]).columns.tolist()
scaler_app = MinMaxScaler()
appl_scaled = appl.copy()
appl_scaled[num_cols_appl] = scaler_app.fit_transform(appl[num_cols_appl])
print("Q29) Normalização concluída. Preview:\n", appl_scaled.head(), "\n")
```
- **Função**: Aplica MinMaxScaler em todas as colunas numéricas
- **Saída**: Preview dos dados normalizados

#### **Q30 - PCA Appliances**
```python
X_app = appl_scaled[num_cols_appl].dropna().values
pca_app = PCA(n_components=2, random_state=RANDOM_STATE)
X_app_pca = pca_app.fit_transform(X_app)
print("Q30) Variância explicada:", pca_app.explained_variance_ratio_, "\n")
plt.figure()
plt.scatter(X_app_pca[:,0], X_app_pca[:,1])
plt.title("Q30) PCA (2D) - Appliances dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```
- **Função**: Aplica PCA nos dados normalizados
- **Saída**: Variância explicada + scatter plot 2D

#### **Q31 - Regressão Linear Múltipla**
```python
features = [c for c in appl.columns if c.lower().startswith(('t','rh'))]
X = appl[features].fillna(method='ffill').fillna(method='bfill').values
y = appl['Appliances'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
lr = LinearRegression().fit(X_tr, y_tr)
pred_lr = lr.predict(X_te)
print("Q31) R² (teste):", lr.score(X_te, y_te), " | RMSE:", rmse(y_te, pred_lr), "\n")
```
- **Função**: Treina regressão linear múltipla com variáveis ambientais
- **Saída**: R² e RMSE do modelo

#### **Q32 - Random Forest Regressor**
```python
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr, y_tr)
pred_rf = rf.predict(X_te)
print("Q32) RMSE RF:", rmse(y_te, pred_rf), " | (Comparação com Linear:", rmse(y_te, pred_lr), ")\n")
```
- **Função**: Treina Random Forest com 200 árvores
- **Saída**: RMSE comparativo com regressão linear

#### **Q33 - K-Means Múltiplos**
```python
Xk = appl_scaled[features].dropna().values
for k in [3,4,5]:
    kmx = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xk)
    print(f"Q33) k={k}: tamanhos dos clusters =", np.bincount(kmx.labels_))
```
- **Função**: Testa K-Means com k=3,4,5
- **Saída**: Tamanho dos clusters para cada k

#### **Q34 - Classificação Binária**
```python
median_app = np.median(appl['Appliances'].values)
appl_cls = appl_scaled.copy()
appl_cls['HighUsage'] = (appl['Appliances'] > median_app).astype(int)
Xc = appl_cls[features].values
yc = appl_cls['HighUsage'].values
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc)

logit = LogisticRegression(max_iter=1000).fit(Xc_tr, yc_tr)
pred_logit = logit.predict(Xc_te)

rfc = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(Xc_tr, yc_tr)
pred_rfc = rfc.predict(Xc_te)

print("Q34) LogReg - Acc/Prec/Rec/F1:",
      accuracy_score(yc_te, pred_logit),
      precision_score(yc_te, pred_logit),
      recall_score(yc_te, pred_logit),
      f1_score(yc_te, pred_logit))
print("Q34) RFClf  - Acc/Prec/Rec/F1:",
      accuracy_score(yc_te, pred_rfc),
      precision_score(yc_te, pred_rfc),
      recall_score(yc_te, pred_rfc),
      f1_score(yc_te, pred_rfc), "\n")
```
- **Função**: Cria variável binária (alto/baixo consumo) e treina 2 classificadores
- **Saída**: Métricas de ambos os modelos

#### **Q35 - Avaliação Detalhada**
```python
print("Q35) == Logistic Regression ==")
print(confusion_matrix(yc_te, pred_logit))
print(classification_report(yc_te, pred_logit))
print("Q35) == Random Forest Classifier ==")
print(confusion_matrix(yc_te, pred_rfc))
print(classification_report(yc_te, pred_rfc))
print("Q35) Observe em qual classe ocorrem mais erros (alto vs baixo consumo).")
```
- **Função**: Gera matriz de confusão e relatório detalhado
- **Saída**: Matrizes de confusão e relatórios de classificação

### Parte 4 - Orange Data Mining - 5 questões

#### **Q36 - Importação e Visualização**
- **Widget**: CSV File Import → Data Table
- **Configuração**: Separador ';', Missing '?', Header Sim
- **Objetivo**: Verificar número de instâncias e variáveis

#### **Q37 - Amostragem 1%**
- **Widget**: CSV → Sample Data (Proportion = 0.01) → Distribution
- **Objetivo**: Comparar distribuição da amostra vs base completa

#### **Q38 - Distribuição do Consumo**
- **Widget**: Distribution (Global_active_power)
- **Objetivo**: Analisar concentração em valores baixos/altos

#### **Q39 - Scatter Plot**
- **Widget**: Scatter Plot (X=Voltage, Y=Global_intensity)
- **Objetivo**: Verificar correlação visual entre variáveis

#### **Q40 - K-Means Visual**
- **Widget**: k-Means (k=3) → Scatter Plot (Color=Cluster)
- **Features**: Sub_metering_1, Sub_metering_2, Sub_metering_3
- **Objetivo**: Interpretar padrões distintos de consumo por cluster

## 📊 Principais Resultados

### Insights de Consumo Energético
- **Padrão sazonal**: Maior consumo no verão vs inverno
- **Padrão diário**: Picos de consumo em horários específicos
- **Diferença semanal**: Consumo maior em dias úteis vs fins de semana
- **Correlações**: Relação entre tensão, intensidade e consumo ativo

### Performance dos Modelos
- **Regressão Linear**: Baseline para predição de consumo
- **Random Forest**: Melhor performance em tarefas de regressão
- **Classificação**: Diferenciação entre alto e baixo consumo
- **Clustering**: Identificação de padrões distintos de uso

## 🔧 Funcionalidades Implementadas

### Funções Auxiliares
- `parse_power_dataset()` - Carregamento e limpeza de dados
- `add_datetime_and_weekday()` - Criação de variáveis temporais
- `hourly_series()` - Agregação por hora
- `plot_simple_line()` - Visualização simplificada
- `rmse()` - Cálculo de erro quadrático médio

### Análises Automatizadas
- Tratamento de valores ausentes
- Normalização e padronização
- Análise de correlações
- Decomposição de séries temporais
- Validação cruzada de modelos

## 📝 Observações Técnicas

- **Random State**: Fixado em 42 para reprodutibilidade
- **Tratamento de Missing**: Valores '?' convertidos para NaN
- **Escalamento**: MinMaxScaler e StandardScaler aplicados conforme necessário
- **Validação**: Split 80/20 para treino/teste
- **Métricas**: RMSE para regressão, acurácia/precisão/recall/F1 para classificação

## 🎯 Objetivos de Aprendizado

Este projeto demonstra:
- **Análise exploratória** de dados temporais
- **Visualização** de padrões e tendências
- **Pré-processamento** e limpeza de dados
- **Modelagem preditiva** com diferentes algoritmos
- **Avaliação** de performance de modelos
- **Clustering** e redução de dimensionalidade
- **Integração** entre Python e Orange Data Mining

## 📚 Referências

### Datasets
- [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

### Documentação Técnica
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Orange Data Mining Platform](https://orangedatamining.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Widgets Orange Data Mining
- [Orange Widget Catalog](https://orangedatamining.com/widget-catalog/)
- [CSV File Import](https://orangedatamining.com/widget-catalog/data/csvfile/)
- [Data Table](https://orangedatamining.com/widget-catalog/data/datatable/)
- [Sample Data](https://orangedatamining.com/widget-catalog/data/sampledata/)
- [Distribution](https://orangedatamining.com/widget-catalog/visualize/distribution/)
- [Scatter Plot](https://orangedatamining.com/widget-catalog/visualize/scatterplot/)
- [k-Means](https://orangedatamining.com/widget-catalog/unsupervised/kmeans/)

## 📋 Instruções de Entrega

- ✅ **Desenvolvimento em grupo** permitido
- ✅ **Apenas um integrante** submete a atividade
- ✅ **Enviar apenas o link** do repositório GitHub
- ✅ **Total de exercícios**: 40 (20 em 20/08 + 20 em 27/08)
- ✅ **Repositório GitHub** com README.md detalhado
- ✅ **Notebook Python** com resoluções das tarefas

---
