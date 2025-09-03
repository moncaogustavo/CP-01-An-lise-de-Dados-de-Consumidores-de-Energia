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

## 📈 Análises Realizadas

### Parte 1 - Exercícios Iniciais (Power Dataset) - 20 questões
1. **Carregamento** e exibição das 10 primeiras linhas
2. **Explicação** da diferença entre Global_active_power e Global_reactive_power
3. **Verificação** de valores ausentes e quantificação
4. **Conversão** de Date para datetime e criação de coluna weekday
5. **Filtro** de 2007 e cálculo da média diária de Global_active_power
6. **Gráfico de linha** da variação de Global_active_power em um dia
7. **Histograma** da variável Voltage e análise da distribuição
8. **Consumo médio** por mês em todo o período
9. **Identificação** do dia com maior consumo de energia ativa
10. **Comparação** entre dias úteis vs fins de semana
11. **Correlação** entre Global_active_power, Global_reactive_power, Voltage e Global_intensity
12. **Criação** da variável Total_Sub_metering (soma das submedições)
13. **Verificação** de meses onde Total_Sub_metering > Global_active_power
14. **Série temporal** do Voltage para 2008
15. **Comparação** entre verão e inverno (hemisfério norte)
16. **Amostragem** de 1% e comparação de distribuições
17. **Normalização** Min-Max das variáveis numéricas
18. **K-means** para segmentar dias em 3 grupos de consumo
19. **Decomposição** de série temporal (6 meses)
20. **Regressão linear** Global_active_power vs Global_intensity

### Parte 2 - Exercícios Adicionais (Power Dataset) - 5 questões
21. **Séries temporais** por hora e identificação de horários de maior consumo
22. **Autocorrelação** em lags de 1h, 24h e 48h
23. **PCA** para redução a 2 componentes principais
24. **Visualização** de clusters no espaço PCA
25. **Regressão polinomial** vs linear (Voltage vs Global_active_power)

### Parte 3 - Appliances Energy Prediction - 10 questões
26. **Carregamento** e inspeção inicial (.info() e .describe())
27. **Distribuição** do consumo (histogramas e séries temporais)
28. **Correlações** com variáveis ambientais (temperatura, umidade)
29. **Normalização** Min-Max dos dados numéricos
30. **PCA** e redução para 2 componentes principais
31. **Regressão Linear Múltipla** para prever Appliances
32. **Random Forest Regressor** e comparação de RMSE
33. **K-Means clustering** com 3 a 5 clusters
34. **Classificação binária** (alto vs baixo consumo baseado na mediana)
35. **Avaliação** com matriz de confusão e métricas (accuracy, precision, recall, F1)

### Parte 4 - Orange Data Mining - 5 questões
36. **Importação** e visualização inicial (CSV File Import + Data Table)
37. **Amostragem** de 1% e comparação de distribuições
38. **Distribuição** do consumo (widget Distribution)
39. **Scatter Plot** Voltage vs Global_intensity
40. **K-Means** com Sub_metering_1/2/3 e visualização de clusters

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
