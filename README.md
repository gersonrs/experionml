# ExperionML

## 💡 **Introdução**

Durante a fase de exploração de um projeto de aprendizado de máquina, o cientista de dados precisa identificar o pipeline mais adequado para o seu problema específico. Esse processo normalmente envolve diversas etapas, como limpeza de dados, criação ou seleção de atributos relevantes, teste de diferentes algoritmos e avaliação de múltiplas configurações de modelos.

Explorar várias combinações de pipelines costuma exigir muitas linhas de código. Quando todo esse processo é realizado em um único notebook, o código rapidamente se torna longo, difícil de manter e pouco organizado. Por outro lado, dividir os experimentos em múltiplos notebooks pode dificultar a comparação entre resultados e a visão geral do progresso do projeto. Além disso, refatorar código para cada novo experimento pode consumir um tempo significativo.

Quantas vezes você já executou as mesmas etapas de pré-processamento para diferentes conjuntos de dados? Quantas vezes precisou copiar e colar código de repositórios antigos para reutilizar em novos projetos?

**ExperionML** foi desenvolvido para resolver esses problemas comuns no fluxo de trabalho de machine learning. A biblioteca atua como uma camada de orquestração sobre todo o pipeline de modelagem, permitindo que cientistas de dados executem experimentos de forma rápida, organizada e reproduzível.

Com o ExperionML, tarefas repetitivas são automatizadas e o foco passa a ser a experimentação e a análise de resultados. Em poucas linhas de código, é possível aplicar etapas essenciais de pré-processamento, selecionar atributos relevantes, treinar múltiplos modelos e comparar seus desempenhos em um mesmo conjunto de dados.

Dessa forma, o ExperionML permite que o usuário avance rapidamente do **dado bruto para insights relevantes**, mantendo os experimentos estruturados e fáceis de analisar.

---

### Exemplo de etapas executadas no pipeline do ExperionML

**1. Limpeza de dados**

- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Detecção e remoção de outliers
- Balanceamento do conjunto de treinamento

**2. Engenharia de atributos**

- Criação de atributos não lineares
- Seleção das variáveis mais relevantes

**3. Treinamento e validação de múltiplos modelos**

- Ajuste de hiperparâmetros
- Treinamento dos modelos no conjunto de treino
- Avaliação no conjunto de teste

**4. Análise dos resultados**

- Cálculo de métricas de desempenho
- Visualizações para comparação entre modelos
