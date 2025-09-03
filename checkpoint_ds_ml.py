# CHECKPOINT 01 – Data Science e Machine Learning no Python e Orange Data Mining
# Gustavo Bispo - R558515
# Gustavo Monção - 557515
# Lucas Barreto - 557107
# Vinicius Murtinho - 551151

POWER_PATH = "household_power_consumption.txt"
APPLIANCES_PATH = "energydata_complete.csv"  
RANDOM_STATE = 42

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

# -------------------- Funções auxiliares --------------------

def parse_power_dataset(path):
    '''
    Carrega o dataset 'Individual Household Electric Power Consumption' do UCI.
    O arquivo padrão é ; separado, com ? para missing.
    Retorna DataFrame com colunas numéricas convertidas e Date/Time em texto (serão tratadas depois).
    '''
    df = pd.read_csv(
        path, 
        sep=';', 
        na_values='?',
        low_memory=False
    )
    # Converte numéricas
    numeric_cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                    'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def add_datetime_and_weekday(df):
    '''
    Cria coluna datetime combinando Date + Time, e coluna Weekday (nome do dia).
    Supõe Date no formato d/m/Y e Time em H:M:S do dataset original.
    '''
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Weekday'] = df['Date_dt'].dt.day_name(locale='en_US')
    df['Year'] = df['Date_dt'].dt.year
    df['Month'] = df['Date_dt'].dt.month
    df['Day'] = df['Date_dt'].dt.day
    return df

def hourly_series(df, value_col='Global_active_power'):
    s = df.dropna(subset=['Datetime']).set_index('Datetime')[value_col].resample('1H').mean()
    return s

def plot_simple_line(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# -------------------- PARTE 1 --------------------

print("\n=== PARTE 1 – POWER ===\n")

# 1
power = parse_power_dataset(POWER_PATH)
print("Q1) Primeiras 10 linhas:\n", power.head(10), "\n")

# 2 (explicação no relatório final – ver enunciado)

# 3
missing_counts = power.isna().sum()
print("Q3) Total de linhas:", len(power))
print("Valores ausentes por coluna:\n", missing_counts, "\n")

# 4
power = add_datetime_and_weekday(power)
print("Q4) Colunas Date_dt e Weekday criadas. Preview:\n", power[['Date','Date_dt','Weekday']].head(), "\n")

# 5
p2007 = power[power['Year'] == 2007].dropna(subset=['Global_active_power', 'Date_dt'])
daily_mean_2007 = p2007.groupby('Date_dt')['Global_active_power'].mean().rename('DailyMeanGAP')
print("Q5) Média diária 2007 – resumo:\n", daily_mean_2007.describe(), "\n")

# 6
one_day = p2007.dropna(subset=['Datetime']).set_index('Datetime').between_time('00:00','23:59')
if not one_day.empty:
    date_example = one_day.index.date[0]
    day_slice = one_day[one_day.index.date == date_example]['Global_active_power']
    plot_simple_line(day_slice.index, day_slice.values,
                     f"Q6) Global_active_power ao longo do dia - {date_example}",
                     "Hora", "kW")
else:
    print("Q6) Não há dados completos para plotar um dia.")

# 7
plt.figure()
power['Voltage'].dropna().hist(bins=50)
plt.title("Q7) Distribuição de Voltage")
plt.xlabel("Voltage (V)")
plt.ylabel("Frequência")
plt.show()
print("Q7) Observação: normalmente concentrada próximo à tensão nominal, com alguma dispersão.\n")

# 8
monthly_mean = power.dropna(subset=['Global_active_power']).groupby(['Year','Month'])['Global_active_power'].mean()
print("Q8) Primeiros meses e estatísticas:\n", monthly_mean.head(), "\n", monthly_mean.describe(), "\n")

# 9
daily_mean_all = power.dropna(subset=['Global_active_power','Date_dt']).groupby('Date_dt')['Global_active_power'].mean()
max_day = daily_mean_all.idxmax()
max_value = daily_mean_all.max()
print("Q9) Dia com maior consumo médio:", max_day, "Valor (kW):", max_value, "\n")

# 10
power['IsWeekend'] = power['Date_dt'].dt.weekday >= 5
weekday_mean = power.loc[power['IsWeekend'] == False].groupby('Date_dt')['Global_active_power'].mean().mean()
weekend_mean = power.loc[power['IsWeekend'] == True].groupby('Date_dt')['Global_active_power'].mean().mean()
print("Q10) Média diária semana (kW):", weekday_mean, " | fim de semana (kW):", weekend_mean, "\n")

# 11
corr_vars = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity']
corr_df = power[corr_vars].dropna().corr()
print("Q11) Correlação entre variáveis:\n", corr_df, "\n")

# 12
for c in ['Sub_metering_1','Sub_metering_2','Sub_metering_3']:
    if c not in power.columns:
        power[c] = np.nan
power['Total_Sub_metering'] = power[['Sub_metering_1','Sub_metering_2','Sub_metering_3']].sum(axis=1, min_count=1)
print("Q12) Total_Sub_metering criada. Preview:\n", power[['Sub_metering_1','Sub_metering_2','Sub_metering_3','Total_Sub_metering']].head(), "\n")

# 13
monthly_tot_sub = power.groupby(['Year','Month'])['Total_Sub_metering'].mean()
monthly_gap = power.groupby(['Year','Month'])['Global_active_power'].mean()
compare = pd.DataFrame({
    'Monthly_Total_Sub_metering': monthly_tot_sub,
    'Monthly_Global_active_power': monthly_gap
}).dropna()
compare['Exceeds'] = compare['Monthly_Total_Sub_metering'] > compare['Monthly_Global_active_power']
print("Q13) Meses em que Total_Sub_metering > Global_active_power (médias mensais):\n", compare[compare['Exceeds']==True].head(), "\n")

# 14
p2008 = power[power['Year'] == 2008].dropna(subset=['Datetime','Voltage'])
if not p2008.empty:
    ts2008 = p2008.set_index('Datetime')['Voltage'].resample('1D').mean()
    plot_simple_line(ts2008.index, ts2008.values, "Q14) Voltage - Média diária (2008)", "Data", "Voltage (V)")
else:
    print("Q14) Não há dados suficientes de 2008 para plotar.")

# 15
gap = power.dropna(subset=['Global_active_power'])
summer = gap[gap['Month'].isin([6,7,8])]['Global_active_power'].mean()
winter = gap[gap['Month'].isin([12,1,2])]['Global_active_power'].mean()
print("Q15) Média GAP - Verão (JJA):", summer, " | Inverno (DJF):", winter, "\n")

# 16
sample = power.sample(frac=0.01, random_state=RANDOM_STATE)
plt.figure()
power['Global_active_power'].dropna().plot(kind='kde')
sample['Global_active_power'].dropna().plot(kind='kde')
plt.title("Q16) Distribuição KDE - Completa vs Amostra 1%")
plt.xlabel("Global_active_power (kW)")
plt.legend(["Completa","Amostra 1%"])
plt.show()

# 17
num_vars = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Total_Sub_metering']
scaled_df = power[num_vars].copy()
scaler = MinMaxScaler()
scaled_vals = scaler.fit_transform(scaled_df.values)
scaled_df = pd.DataFrame(scaled_vals, columns=num_vars, index=power.index)
print("Q17) Exemplo de dados escalados (Min-Max):\n", scaled_df.head(), "\n")

# 18
daily = power.dropna(subset=['Date_dt']).groupby('Date_dt')[['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Total_Sub_metering']].mean().dropna()
km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
daily['cluster'] = km.fit_predict(StandardScaler().fit_transform(daily[['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Total_Sub_metering']]))
print("Q18) Tamanho dos clusters (dias):\n", daily['cluster'].value_counts(), "\n")
print("Q18) Perfil médio por cluster:\n", daily.groupby('cluster').mean(), "\n")

# 19
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
        else:
            print("Q19) Janela de 6 meses insuficiente para decomposição.")
    else:
        print("Q19) Série vazia para decomposição.")
else:
    print("Q19) Pulando: statsmodels não instalado.")

# 20
df_lr = power[['Global_active_power','Global_intensity']].dropna()
X = df_lr[['Global_intensity']].values
y = df_lr['Global_active_power'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
linreg = LinearRegression().fit(X_train, y_train)
pred = linreg.predict(X_test)
print("Q20) RMSE:", rmse(y_test, pred))
print("Coeficiente:", linreg.coef_[0], "Intercepto:", linreg.intercept_, "\n")

# -------------------- PARTE 2 --------------------
print("\n=== PARTE 2 – POWER (adicionais) ===\n")

# 21
s_hourly = power.dropna(subset=['Datetime']).set_index('Datetime')['Global_active_power'].resample('1H').mean().dropna()
hourly_pattern = s_hourly.groupby(s_hourly.index.hour).mean()
print("Q21) Top 5 horários de maior consumo médio:\n", hourly_pattern.sort_values(ascending=False).head(5), "\n")
plt.figure()
plt.plot(hourly_pattern.index, hourly_pattern.values)
plt.title("Q21) Padrão médio por hora do dia - Global_active_power")
plt.xlabel("Hora do dia")
plt.ylabel("kW (média)")
plt.show()

# 22
def autocorr_at_lag(series, lag):
    return series.autocorr(lag=lag)
for lag in [1, 24, 48]:
    print(f"Q22) Autocorrelação (lag={lag}h):", autocorr_at_lag(s_hourly, lag))
print("Q22) Autocorrelações altas em 24/48h indicam padrão diário repetido.\n")

# 23
sel = power[['Global_active_power','Global_reactive_power','Voltage','Global_intensity']].dropna()
X_std = StandardScaler().fit_transform(sel.values)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_std)
print("Q23) Variância explicada:", pca.explained_variance_ratio_, "\n")

# 24
km_pca = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit(X_pca)
labels = km_pca.labels_
plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
plt.title("Q24) PCA (2D) + KMeans (3 clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
print("Q24) Verifique visualmente a separação dos grupos.\n")

# 25
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
order = np.argsort(X_test.squeeze())
plt.figure()
plt.scatter(X_test[order], y_test[order])
plt.plot(X_test[order], pred_lin[order])
plt.plot(X_test[order], pred_poly[order])
plt.title("Q25) Ajuste: Linear vs Polinomial (g2)")
plt.xlabel("Voltage")
plt.ylabel("Global_active_power")
plt.legend(["Dados","Linear","Polinomial (g2)"])
plt.show()

# -------------------- PARTE 3 – Appliances --------------------
print("\n=== PARTE 3 – Appliances Energy Prediction ===\n")

# 26
appl = pd.read_csv(APPLIANCES_PATH)
print("Q26) .info():")
print(appl.info())
print("Q26) .describe():\n", appl.describe(include='all'), "\n")

# 27
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
print("Q27) Avalie visualmente se há concentração em valores baixos/altos.\n")

# 28
cand_cols = [c for c in appl.columns if c.lower().startswith(('t','rh'))]
corr_cols = ['Appliances'] + cand_cols
corr_appl = appl[corr_cols].corr()['Appliances'].sort_values(ascending=False)
print("Q28) Correlações (Appliances vs variáveis ambientais):\n", corr_appl, "\n")

# 29
num_cols_appl = appl.select_dtypes(include=[np.number]).columns.tolist()
scaler_app = MinMaxScaler()
appl_scaled = appl.copy()
appl_scaled[num_cols_appl] = scaler_app.fit_transform(appl[num_cols_appl])
print("Q29) Normalização concluída. Preview:\n", appl_scaled.head(), "\n")

# 30
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

# 31
features = [c for c in appl.columns if c.lower().startswith(('t','rh'))]
X = appl[features].fillna(method='ffill').fillna(method='bfill').values
y = appl['Appliances'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
lr = LinearRegression().fit(X_tr, y_tr)
pred_lr = lr.predict(X_te)
print("Q31) R² (teste):", lr.score(X_te, y_te), " | RMSE:", rmse(y_te, pred_lr), "\n")

# 32
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr, y_tr)
pred_rf = rf.predict(X_te)
print("Q32) RMSE RF:", rmse(y_te, pred_rf), " | (Comparação com Linear:", rmse(y_te, pred_lr), ")\n")

# 33
Xk = appl_scaled[features].dropna().values
for k in [3,4,5]:
    kmx = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xk)
    print(f"Q33) k={k}: tamanhos dos clusters =", np.bincount(kmx.labels_))
print()

# 34
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

# 35
print("Q35) == Logistic Regression ==")
print(confusion_matrix(yc_te, pred_logit))
print(classification_report(yc_te, pred_logit))
print("Q35) == Random Forest Classifier ==")
print(confusion_matrix(yc_te, pred_rfc))
print(classification_report(yc_te, pred_rfc))
print("Q35) Observe em qual classe ocorrem mais erros (alto vs baixo consumo).")

# -------------------- PARTE 4 – Orange (instruções) --------------------
print("\n=== PARTE 4 – Orange Data Mining (passo a passo) ===\n")
from textwrap import dedent as _ded
print(_ded('''
36) CSV File Import -> Data Table
   - Arquivo: household_power_consumption.txt
   - Separador: ;   | Missing: ? | Header: Sim
   - Data Table: veja no topo quantas instâncias e variáveis.

37) Sample Data (1%)
   - CSV -> Sample Data (Proportion = 0.01) -> Distribution (Global_active_power).
   - Compare com a Distribution da base completa.

38) Distribution
   - Inspecione Global_active_power: concentração em valores baixos ou cauda alta?

39) Scatter Plot
   - X = Voltage, Y = Global_intensity. Verifique se há correlação visível.

40) k-Means (k=3)
   - Features: Sub_metering_1, _2, _3.
   - k-Means -> Scatter Plot (Color = Cluster).
   - Interprete padrões: perfis distintos de consumo por cluster.
'''))
