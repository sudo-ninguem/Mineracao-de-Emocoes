######################################## MINERADOR DE EMOÇÕES ############################################

'''
A ideia neste programa é fazer um minerador de emoções conforme o banco de dados com frase abaixo
'''

#################################### BIBLIOTECA ########################################

import nltk


'''
A biblioteca (nltk) é uma das principais bibliotecas para linguagem natural em Python. 
Neste caso é bom lembrar que essa bibliotéca não é NATIVA do python tivemos que primeiramente usar um
sudo pip install nltk
'''


#################################### CRIANDO O BANCO DE DADO DE FRASES ##############################

base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')] 

'''
Aqui nos estamos criando uma base de dado com frases para treinar o nosso modelo. 
Repare que vamos ter a frase, que poderia se dizer ser os ATRIBUTOS e vamos ter logo em seguida, após a 
virgula a emoção, que poderiamos dizer se tratar da CLASSE
'''


##################################### REMOÇÃO DE PALAVRAS DESNECESSÁRIAS ##################################

'''
Neste nosso exemplo nem precisária, mas no mundo real teriamos DEZENAS de frases e isso acaba prejudicando
o processamento do nosso sistema, sendo assim sempre devemos remover palavras desnecessárias (stopwords)
para que assim nosso banco de dados fique o mais OTIMIZADO POSSÍVEL
'''
nltk.download('stopwords') ## Devemos primeiramente baixar as (stopwords)

stopwords = nltk.corpus.stopwords.words('portuguese')

'''
A função (nltk.corpus.stopwords.words) vai trazer uma lista de palavras DESNECESSÁRIAS para classificação
de frases, palavras como (artigo, verbo de ligação, etc.) No caso essa função vai trazer a lista dessas
palavras desnecessárias de acordo com cada idioma, e no nosso caso é o (portuguese)
'''

############################### CRIANDO UMA FUNÇÃO PARA REMOVER AS STOPWORDS#######################
'''
Uma vez que temos as listas das palavras consideradas como desnecessárias vamos criar uma função
para percorrer o banco de dados apagando essas palavras desnecessárias
'''
def remove_stopwords(texto): ## O parametro que essa função recebe é justamente nosso banco de dados
    frase = [] ## Criando uma lista vazia para guardar o banco de dados após a remoção das stopwords
    for (palavra, emocao) in texto:
        semstop = [p for p in palavra.split() if p not in stopwords]
        frase.append((semstop, emocao))
    return frase


'''
Dentro da váriavel (semstop) criamos um outro laço for com (p) que vai percorrer cada (palavra)
e usano o método nativo (split) as frases serão DIVIDIDAS A CADA ESPAÇO, mantendo assim somente a palavra
e caso a palavra (if) seja diferente de  (stopwords) ai sim vamos ARMAZENAR na váriável (sempstop)

por fim vamos fazer um append na lista (frase) para adicionar as palavras após todo o tratamento de dados
(semstop) e a sua classe (emocao).
'''

#### Fazendo um teste para ver como ficou

print(remove_stopwords(base)) ## Perceba que ficou somente as palavras MAIS IMPORTANTES e sem espaço. 


########################################## REMOVENDO O RADICAL DAS PALAVRAS #############################

nltk.download('rslp') ## Temos que baixar esses dados para usar a função (nltk.stem.RSLPStemmer)

'''
Para otimizar ainda mais nosso sistema vamos criar uma função que vai retirar o radical das palavras
deixando apénas as letras realmente nescessárias 
'''
def aplicastemme(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frase_sem_stemmer = []
    for (palavra, emocao) in texto:
        frase_com_stemmer = [str(stemmer.stem(p)) for p in palavra.split() if  p not in stopwords]
        frase_sem_stemmer.append((frase_com_stemmer, emocao))
    return frase_sem_stemmer


'''
EXPLICANDO A LINHA DE CÓDIGO SUPERIO 

A função responsável por tirar o radical das palavras é a função (nltk.stem.RSLPStemmer()) entretanto
para evitar de ficar copiando seu nome toda hora jogamos essa função dentro da váriavel (stemmer)

Criamos uma nova lista chamada (frase_sem_stemmer) para receber justamente as frases do nosso banco 
de dados sem o radical 

Fizemos um laço for, percorrendo (palavra, emocao) no parametro (texto) que vai ser nosso banco de dados

e criamos a váriavl (frase_com_stemmer) essa frase vai receber a função (stemmer.stem) passando como 
parametro (p) que vai fazer um for em cada (palavra) EXCLUINDO OS ESPAÇOS (split).
Sendo assim graças a função (stemmer.stem) ao percorrer cada palavra fai ser retirado o seu radical 
e o espaço. 

Por fim novamente dizemos que SÓ VAI ARMAZENAR se (if) aquela palavra (p) for DIFERENTE (not in) da lista
das stopwords.

Percebaw então que na mesma linha de código colocamos como requisito: 
    
RETIRAR O RADICAL (stemmer.stem)
RETIRAR O ESPAÇO (split())
EXCLUIR AS PALAVRAS DESNECESSÁRIAS (if  p not in stopwords)
'''

###################################### APLICANDO A FUNÇÃO A NOSSA BASE DE DADOS ###########################

frase_radical = aplicastemme(base)

'''
Se abrirmos agora a variavel (frase_radical) vamos ver que não teremos (espaço) nem palavras 
desnecessárias e agora as palavras estão apénas com o RADIAL

Esse deve ser o padão para usarmos de banco de dados, ou seja, tudo que fizemos até aqui foi tratamento
de dados
'''

###################################### FUNÇÃO BUSCA PALAVRAS ########################################

def buscapalavra(frase):
    todaspalavras = []
    for (palavra, emocao) in frase:
        todaspalavras.extend(palavra)
    return todaspalavras

'''
Nos agora vamos criar uma função que vai receber um parametro, no caso vai ser nosso banco de dados JA 
TRATADO (frase_radical) e dentro deste banco de dados vai ser gerado uma lista APENAS COM AS PALAVRAS.

Ou seja, a ideia é separar as palavras e deixar somente elas 
'''

#################################### APLICANDO A FUNÇÃO NO NOSSO BANCO DE DADOS #######################


palavra = buscapalavra(frase_radical)

## Vamos aplicar a função de (buscapalavra) e guardar a lista contendo as palavras na váriavel palavra


############################# FUNÇÃO FREQUENCIA DE PALAVRAS #########################################

def buscafrequencia (palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

'''
Vamos buscar o número de vezes que uma palavra se repete no nosso banco de dados para podermos eliminar
e reduzir ainda mais o processamento de dados na hora de gerar nosso modelo. 

A propria biblioteca (nltk) tem um método especifico para buscar frequencia de palavras em um banco
de dados que é justamente a (FreqDist) recebendo como parametro as palavras
'''


############################################### BUSCANDO A FREQUENCIA DE PALAVRAS #########################

frequencia = buscafrequencia(palavra)
'''
Quando executarmos a função o numero de palavras vai de (56) para (43) retirando justamente as palavras
repetidas

'''

###################################### FUNÇÃO DE PALAVRAS UNICAS ############################

'''
O método (FreqDist) retorna um discionário em que a chave é as palavras e os valores são os numeros de 
vezes que aquela palavra repetiu. Para nós só é importante as palavras em si e não o número de repetições

Sendo assim vamos fazer uma função agora para pegar apenas as chaves do dicionário, ou seja, as palavras
propriamente dito
'''

def palavrachave(palavras):
    chave = palavras.keys() ## Pegando só a chave dos dicionários 
    return chave

####################################### PEGANDO AS PALAVRAS UNICAS ###############################


palavras_unicas = palavrachave(frequencia)


'''
Agora temos um dicionário contendo apenas AS PALAVRAS ESSENCIAIS, NO RADICAL, SEM REPETIÇÃO 
'''

###################################### BOOLEANO #################################################

'''
As palavras do nosso banco dedados, como falamos anteriormente, funciona como se fosse nossos atributos
E agora nos temos que fazer um modelo que analise se determinada palavra existe caso exista ele vai
colocar TRUE caso não ele coloca FALSE. 

O nosso modelo vai percorrer todo o nosso banco de dados colocando True ou False. Isso vai ajudar a mapear
as palavras relacionadas com a classe, sendo assim quando o modelo se deparar com uma nova frase
ele vai analisar a quantidade de TRUE ou FALSE para ver se aquela nova frase se parece mais com qual 
tipo de classe (sentimento)

'''

################################ CRIANDO UMA FUNÇÃO GERANDO UM BOOLEANO ######################

def extratorpalavras (frases):
    extrator = set(frases)
    caracteristicas = {}
    for p in palavras_unicas: ## estamos percorrendo a váiavel (palavras unicas)
        caracteristicas["%s" % p] = (p in extrator)
    return caracteristicas


'''
Nesta função estamos criando 2 váriaveis uma que vai receber os elementos do parametro (extrator)
e uma outra váriavel que vai receber um dicionário vazio (caracteristicas)

Apos isso fazemos um for e a variavel (caracteristica) so vai receber um valor de (true) se a palavra 
percorrida (p) for igual a que pegamos no parametro (extrator)
'''



#################################### TESTE #########################################



teste = extratorpalavras(['tim', 'gol', 'nov'])

'''
Nesse teste geramos algumas palavras já processadas apénas para teste e repare que a váriavel 
ira gerar uma loista de (false) só sendo tru na palavra (nov), pois ela existe na nossa lista 
(palavras_unicas)
'''

###################################### GERANDO O MODELO ####################################3

'''
Feito tudo agora vamos gerar um modelo com a função (extrator) e o nosso banco de dados COM A CLASSE, 
ou seja, nos não vamos usar a lista de palavras unicas, pois precisamos que nosso modelo apos converter
os valores em True ou False veja qual é a classe dominante
'''

modelo = nltk.classify.apply_features(extratorpalavras, frase_radical)

'''
Para gerar o modelo podemos usar o método (classify.apply_features) que é nativo da biblioteca (nltk)
esse esse método vai receber como parametro uma função (no caso vai ser a função de extração) e uma lista
que no caso vai ser (frase_radical) pois aqui vamos ter AS PRINCIPAIS PALAVRAS, NO RADICAL E A SUA CLASSE
'''

print(modelo[0])

'''
Se fizermos um print na primeira posição do nosso modelo aonde teria a frase 
('eu sou admirada por muitos','alegria') foi gerado um discionário com (true) para as palavras

1- admir
2 - muit

E false para o resto das palavras sendo que no final desse dicionário vamos ter a classe (alegria)

Sendo assim quando passarmos palavras novas o modelo vai ver a incidencia dessas palavras (ex admir,muit)
E dizer se a frase é de alegria ou tristesa com base na quantidade de palavras de determinada classe
'''



############################################# GERANDO O CLASSIFICADOR NAVE BAISE ####################################

'''
O (nltk) utiliza como algoritimo de aprendizado o Naive Baise, e vamos usar ele agora para gerar um classificador
'''

classificador = nltk.NaiveBayesClassifier.train(modelo)

'''
Sendo assim agora estamos criando um classficiador (com base no modelo) que vai analisar as probabilidades
e dizer a provavel classe de futuras palavras
'''


#########################################  CRIANDO UMA NOVA FRASE FRASES ######################################

'''
Agora que geramos o classificador conforme nosso modelo vamos testar algumas frases para ver se nosso
classificador consegue acertar as emoções
'''
frase1 = "Tenho medo da prova"

### Essa frase está no modo bruto, sendo assim devemos fazer o preprocessamento dela para o (classificador)

processamento = []

processado = nltk.stem.RSLPStemmer() ## estamos colocando a função (nltk.stem.RSLPStemmer) dentro da váriavel

for (palavra) in frase1.split():
    contem = [p for p in palavra.split()]
    processamento.append(str(processado.stem(contem[0])))

'''
Aqui nos estamos percorrendo a frase (frase1) em cada elemento e processando ela para gerar APENAS OS RADICAIS
como fizemos anteriormente na hora de gerar o modelo.

Aqui nos só vamos colocar a frase em radical, NÃO PRECISAMOS REMOVER as palavras desnecessárias (stopwords)
pois o nosso modelo já não tem, sendo assim ele simplesmente vai ignorar essas (stopwords) das frases
'''

################################## ANALISANDO OS SENTIMENTOS DA NOVA FRASE #####################################


sentimentos_frase1 = extratorpalavras(processamento)
print (classificador.classify(sentimentos_frase1))



############################## 2 FRASE ########################################################3


frase2 = "O amor que tenho pro você é insubstituível"

processamento = []

processado = nltk.stem.RSLPStemmer()
for (palavra) in frase2.split():
    contem = [p for p in palavra.split()]
    processamento.append(str(processado.stem(contem[0])))


sentimentos_frase2 = extratorpalavras(processamento)
print (classificador.classify(sentimentos_frase2))


