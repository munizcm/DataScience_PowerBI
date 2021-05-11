##########################################################################
#
#            Microsof Power BI Para Data Science 2.0
#          
#                       Data Science Academy
#
#                          Mini-projeto 03
#
# Prevendo a Inadimplência de Clientes com Machine learning e Power BI
#
##########################################################################

#Definindo a área de trabalho
setwd("~/OneDrive/PowerBI_DataScience/Cap.15")
getwd() #Conferindo o diretório
dir()   #Conferindo os arquivos que existem na minha pasta

#Definição do problema 
#Está no manual do pdf 

#Instalando os pacotes para o projeto 
#install.packages("Amelia")
#install.packages("caret")
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("reshape")
#install.packages("randomForest")
#install.packages("e1071")

#Carregando os pacotes 
library(Amelia)         #Usado para tratar dados ausentes
library(caret)          #Permite construir modelos de machine learning e processar dados
library(ggplot2)        #Construir gráficos  
library(dplyr)          #Tratar e manipular dados
library(reshape)        #Modificar o formato dos dados
library(randomForest)   #Trabalhar com machine learning
library(e1071)          #Trabalhar com machine learning

#Carregando o banco de dados 
dados_clientes<- read.csv("dataset.csv")

#Visualizando os dados e sua estrutura
View(dados_clientes)    #Vizualizar a tabela
dim(dados_clientes)     #Dimensao do nosso banco de dados (linhas e colunas)
str(dados_clientes)     #Mostra a lista de variáveis que o data frame apresenta (mostra o tipo de cada uma delas)
summary(dados_clientes) #Apresenta o resumo estatístico das variáveis

############ Análise exploratória, limpeza e transformação ################

#Remobendo a primeira coluna ID 
dados_clientes$ID<- NULL
dim(dados_clientes)
View(dados_clientes)

#Renomeando a coluna de classe 
colnames(dados_clientes)
colnames(dados_clientes)[24]<- "inadimplente"
colnames(dados_clientes)

#Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))             #Retorna a quantidade de NA encontrada em cada uma das colunas
?missmap  
missmap(dados_clientes, main = "Valores Missing Observados")  #Um gráfico que retorna a presença de dados ausentes
dados_clientes<- na.omit(dados_clientes)                      #Retirar as observações que possuem NA

# Convertendo os atributos genero, escolaridade, estado civil e idade para 
# fatores (categorias)

colnames(dados_clientes)
colnames(dados_clientes)[2]<- "Genero"
colnames(dados_clientes)[3]<- "Escolaridade"
colnames(dados_clientes)[4]<- "Estado_civil"
colnames(dados_clientes)[5]<- "Idade"
View(dados_clientes)

#Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
#A funcao cut converte variaveis numericas para fator
dados_clientes$Genero<- cut(dados_clientes$Genero,
                            c(0,1,2),
                            labels = c("Masculino",
                                       "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

#Escolaridade
View(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade<- cut(dados_clientes$Escolaridade,
                                  c(0,1,2,3,4),
                                  labels = c("Pos Graduado",
                                             "Graduado",
                                             "Ensino Medio",
                                             "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

#Estado civil
View(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)
dados_clientes$Estado_civil<- cut(dados_clientes$Estado_civil,
                                  c(-1,0,1,2,3),
                                  labels = c("Desconhecido",
                                             "Casado",
                                             "Solteiro",
                                             "Outro"))
View(dados_clientes$Estado_civil)
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)

#Convertendo a variável para o tipo fator com faixa etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)
dados_clientes$Idade<- cut(dados_clientes$Idade,
                           c(0,30,50,100),
                           labels = c("Jovem",
                                      "Adulto",
                                      "Idoso"))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)

# Convertendo a variável que indica pagamentos para o tipo de fator

dados_clientes$PAY_0<- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2<- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3<- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4<- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5<- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6<- as.factor(dados_clientes$PAY_6)

#Dataset após as conversoes
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores NAs observados")
dados_clientes<- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores NAs observados")
dim(dados_clientes)

#Alterando a variável dependente para o tipo fator
str(dados_clientes$inadimplente)
dados_clientes$inadimplente<- as.factor(dados_clientes$inadimplente)

#Total de inadimplente versos nao inadimplente
?table
table(dados_clientes$inadimplente)

#Vejamos a porcentagem entre as classes
prop.table(table(dados_clientes$inadimplente))

#Plot da distribuuicao usando ggplot2
par(mfrow = c(1,1))
qplot(inadimplente, data=dados_clientes, geom = "bar")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set seed: É utilizado pois vamos fazer a divisao dos dados em treino
# e teste e esse é um processo randomico. O set.seed é só para termos os mesmos resultados
# da aula
set.seed(12345)

#Amostragem estratificada
#Seleciona as linhas de acordo com a variavel inadimplente com strata
?createDataPartition
indice<- createDataPartition(dados_clientes$inadimplente, p = 0.75,
                             list = FALSE) #75% dos dados vao para treino e 25% para o teste
dim(indice)

#Definimos os dados de treinamento como sbconjunto do conjunto de dados original
# com números de indices de linha (conforme indentificado acima) e todas as colunas

dados_treino<- dados_clientes[indice,]
table(dados_treino$inadimplente)

#Verificar a porcentagem entre classes. Ela deve seguir a mesma proporcao dos dados completos
prop.table(table(dados_treino$inadimplente))

#Comparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados<- cbind(prop.table(table(dados_treino$inadimplente)),
                      prop.table(table(dados_clientes$inadimplente)))
colnames(compara_dados)<- c("Treinamento", "Original")
compara_dados

#Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados<- melt(compara_dados)
melt_compara_dados

#Plot para ver a distribuição do treinamento vs. original 
ggplot(melt_compara_dados, aes(x = X1, y = value))+
  geom_bar(aes(fill=X2), stat = "identity", position = "dodge")+
  theme(axis.title.x = element_text(angle= 90, hjust = 1))

# Tudo o que não está no dataset do treinamento está no dataset de teste, Observe o sinal -(menos)
# Isso porque não podemos misturar os dados teste do treino, assim selecionamos todos os 
# dados menos aqueles que estao contidos no indice
dados_teste<- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)


############ Modelo de Machine Learning ################

#Construindo a primeira versao do modelo
?randomForest
modelo_v1<- randomForest(inadimplente ~ . , data = dados_treino)
modelo_v1

#Avaliando o modelo
plot(modelo_v1)
# A gente pode observar que o modelo começa mal, mas ele melhora com o tempo
# Em cima é a performace do modelo, ela começa ruim e vai melhorando
# Na parte de baixo é o erro do modelo, ele começa com o erro alto e vai aprendendo

#Previsões com dados de teste
previsoes_v1<-predict(modelo_v1, dados_teste) 

#Confusion Matrix - Uma forma de avaliar o modelo
?caret::confusionMatrix
cm_v1<- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1
# A acurácia a cima de 75% é tida como excelente, a baixo de 50% não é tolerado,
# Entre 50% e 70% deve-se fazer melhorias
# A acurácia é uma metrica global para avaliacao do modelo preditivo

# Calculando Precision, Recall e F1-Score, métricas locais de avaliação do modelo preditivo

y<- dados_teste$inadimplente
y_pred_v1<- previsoes_v1

precision<- posPredValue(y_pred_v1, y)
precision

recall<- sensitivity(y_pred_v1, y)
recall

F1<- (2 * precision*recall)/(precision+recall)
F1

# Balanceamento de classe
# Para balancear as classes podemos utilizar duas técnicas: oversampling ou undersampling
# Vamos utilizar a técnica oversampling
install.packages("DMwR")
library(DMwR)
?SMOTE

# Aplicando o SMOTE - SMOTE : Synthetic Minitory Over - sampling Techniquue
# http://arxiv.org/pdf/1106.1813.pdf
table (dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)
dados_treino_bal<- SMOTE(inadimplente ~ ., data = dados_treino)
table(dados_treino$inadimplente)
prop.table(table(dados_treino_bal$inadimplente))

# Construindo a segunda versao do modelo- utilizando os dados balanceados 
modelo_v2<- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2

#Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
# OBS : Não faz sentido aplicar o balanceamento em dados de teste
# Isso porque o balanceamento é feito para treinar o modelo, uma vez que o modelo aprendeu
# pode-se aplicar, na teoria, qualquer dado teste. 
previsoes_v2<-predict(modelo_v2, dados_teste) 

# Confusion Matrix - Uma forma de avaliar o modelo
?caret::confusionMatrix
cm_v2<- caret::confusionMatrix(previsoes_v2, dados_teste$inadimplente, positive = "1")
cm_v2
# Aqui podemos observar que a acuracia do modelo reduziu, porém ele é tido como um modelo muito melhor que o V1
# O V1 tem uma acurácia que mascara um dos problemas do modelo - que é um modelo desequilibrado (classes desbalanceada)
# Não adianta só olhar para a acurácia

# Calculando Precision, Recall e F1-Score, métricas locais de avaliação do modelo preditivo

y<- dados_teste$inadimplente
y_pred_v2<- previsoes_v2

precision<- posPredValue(y_pred_v2, y)
precision

recall<- sensitivity(y_pred_v2, y)
recall

F1<- (2 * precision*recall)/(precision+recall)
F1

# Aqui podemos observar que as métricas calculadas estão equilibradas
# Apesar da acurácia do modelo v1 ser maior, ao inves dele ter um desempenho bom pra ambas as classes (pagou e nao pagou),
# Ele tinha um desempenho muito bom pra uma, e ruim pra outra
# O modelo v2 é um modelo melhor pois acerta mais as duas classes.
# Porém a v2 ainda não é uma versao ideal ela é apenas uma versao mais equilibrada, existem inumeras possibilidades para melhorar o modelo


# Nas versoes anteriores utilizamos TODAS as variáveis preditoras que tinhamos. Nessa nova versao vamos utilizar apenas algumas 
# Isso porque algumas dessas variáveis podem não ter relacao com o que queremos avaliar (não ser relevante)

# Assim, valos avaliar a importancia das variaveis preditoras para as previsoes
View(dados_treino_bal)
varImpPlot(modelov2)

#Obtendo as variaveis mais importantes
imp_var <- varImpPlot(modelov2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[, "MeanDecreaseGini"],2))

# Criando o rank de variaveis baseado na importancia 
rankImportance <- varImportance%>%
  mutate(Rank = paste0("#", dense_rank(desc(importance()))))

# Usando ggplot2 para visualizar a importancia relativa das variáveis
ggplot(rankImportance, 
       aes(x = reorder(Variables, Importance),
           y = Importance, 
           fill = Importance)) + 
  geom_bar(stat = "identity") + 
  geom_text(aes(x = variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = "red")+
  labs (x = "variables") + 
  coord_flip

# Aqui, podemos observar que a variável PAY_0 é a mais importante para a construção do modelo, 
# já as variáveis, escolaridade, idade, estado civil e gênero, parecem não ter muita relevância
# Assim, vamos remover algumas variáveis no próximo modelo. 

# Criando a terceira versão do nosso modelo
# A decisao de quais variaveis vao ser utilizadas no modelo é uma decisao do Cientista de dados
colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_bal)
modelo_v3

# Não há garantia que a V3 será melhor, isso são tentativas de melhorar o modelo

#Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
# OBS : Não faz sentido aplicar o balanceamento em dados de teste
# Isso porque o balanceamento é feito para treinar o modelo, uma vez que o modelo aprendeu
# pode-se aplicar, na teoria, qualquer dado teste. 
previsoes_v3<-predict(modelo_v3, dados_teste) 

# Confusion Matrix - Uma forma de avaliar o modelo
?caret::confusionMatrix
cm_v3<- caret::confusionMatrix(previsoes_v3, dados_teste$inadimplente, positive = "1")
cm_v3

# Aqui podemos ver que conseguimos um pequeno ganho no nosso valor de acurácia, e a nossa matriz de confusão 
# melhorou um pouquinho.

# Calculando Precision, Recall e F1-Score, métricas locais de avaliação do modelo preditivo

y<- dados_teste$inadimplente
y_pred_v3<- previsoes_v3

precision<- posPredValue(y_pred_v3, y)
precision

recall<- sensitivity(y_pred_v3, y)
recall

F1<- (2 * precision*recall)/(precision+recall)
F1

# Essas outras métricas também melhoraram um pouco. Porém vale ressaltar que qualquer ganho é bom. 

# Após encontrar um modelo ideal, temos que salvá-lo em disco. Isso porque muitas vezes o modelo leva muito tempo para ser treinado
# Se não salvarmos, temos que rodar ele novamente. Para isso utilizamos a função a baixo :

# Tenho que rodar o modelo1 porque tive problemas com o pacote da funcao SMOTE
# Por isso vou fazer essa mudança 

modelo_v4 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino)
modelo_v4

saveRDS(modelo_v4, file = "modelo/modelo_v4.rds")

#Carregando o modelo 
modelo_final<- readRDS("modelo/modelo_v4.rds")


# Previsoes com novos dados de 3 clientes 

# Dados dos novos clientes 

PAY_0 <- c(0,0,0)
PAY_2<- c(0,0,0)
PAY_3 <- c(1,0,0)
PAY_AMT1<- c(1100,1000,1200)
PAY_AMT2<- c(1500,1300,1150)
PAY_5 <- c(0,0,0)
BILL_AMT1<- c(350,420,280)

# Concatenando em um dataframe
novos_clientes<- data.frame(PAY_0, PAY_2, PAY_3,PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

#Previsoes
previsoes_novos_cliente<- predict(modelo_final, novos_clientes)

# Checando os tipos de dados 
str(dados_treino)
str(novos_clientes)

#Convertendo para fator
novos_clientes$PAY_0<- factor(novos_clientes$PAY_0, levels = levels(dados_treino$PAY_0))
novos_clientes$PAY_2<- factor(novos_clientes$PAY_2, levels = levels(dados_treino$PAY_2))
novos_clientes$PAY_3<- factor(novos_clientes$PAY_3, levels = levels(dados_treino$PAY_3))
novos_clientes$PAY_5<- factor(novos_clientes$PAY_5, levels = levels(dados_treino$PAY_5))


#Previsoes
previsoes_novos_cliente<- predict(modelo_final, novos_clientes)
View(previsoes_novos_cliente)

#Nenhum cliente novo vai ficar inadimplente :D
