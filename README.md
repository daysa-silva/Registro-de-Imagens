# MS777 - Projeto Supervisionado

A disciplina de MS777 é uma eletiva do Depertamento de Matemática Aplicada do IMECC que presta-se a gerar créditos pelo desenvolvimento de um projeto de pesquisa sob a orientação de um docente da UNICAMP, implicando obrigatoriamente na produção de uma monografia e participação em sessão de apresentação dos resultados perante uma banca ao final do semestre.

# Registro de Imagens: Técnicas e Aplicações

## Introdução

Registro de imagens é o processo de correspondência ou alinhamento entre duas ou mais imagens capturadas da mesma cena, porém, obtidas por diferentes sensores, em diferentes instantes de tempo ou sob diferentes pontos de observação.

Na prática, o de registro de imagens é fundamental no processamento e análise de imagens, sendo uma etapa indispensável em problemas como fusão de imagens, reconhecimento de padrões, segmentação de imagens e classificação de imagens.

Dada duas imagens, o problema de registro de imagens é alinhar a primeira imagem *imagem detectada* na segunda imagem, a *imagem de referência* a partir de uma transformação espacial.

Neste trabalho dividiremos o processo de registro de imagens em 4 etapas:

- Reconhecer diferentes características das imagens a partir de diferentes algoritmos.

- Combinar as características correspondentes em várias imagens.

- Computar a matriz de homografia usando inliers ou correspondências positivas verdadeiras.

- Fundir a imagem detectada com a imagem de referência e realizar a mesclagem entre elas.