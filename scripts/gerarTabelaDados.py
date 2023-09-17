import logging
import time
import cv2
import numpy as np
import pandas as pd

def detectAndDescribe(image, method):
    '''
    Calcula os ponto-chaves e características usando um especifíco método
    '''

    assert method in ['orb','brisk', 'surf', 'sift'], 'Você precisa definir um método de detecção. Valores são: "sift", "brisk", "surf" e "orb"'

    if method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()

    inicio = time.time_ns()
    (kps, features) = descriptor.detectAndCompute(image, None)
    fim = time.time_ns()
    
    
    tempo =  (fim - inicio) * 1e-9

    return (kps, features, tempo)

def matchKeyPoints(featuresA, featuresB, method, feature_extractor, ratio=0.75,):
    "Cria e retorna as correspondências entre os pontos-chaves das duas imagens"
    
    crossCheck = True if method == 'bf' else False

    # Criar o matcher correspondente ao método utilizado para extrair as características
    if feature_extractor == 'sift' or feature_extractor == 'surf':
        logging.debug('Usando a norma L2 para o sift ou surf')
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif feature_extractor == 'orb' or feature_extractor == 'brisk':
        logging.debug('Usando a norma Hamming para o orb ou brisk')
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)

    # Encontrar os pares correspondentes do
    inicio = time.time_ns()
    if method == 'bf':
        best_matches = bf.match(featuresA,featuresB)
        matches = sorted(best_matches, key = lambda x:x.distance)
        
    elif method == 'knn':
        best_matches = bf.knnMatch(featuresA,featuresB, 2)
        
        matches = []
        for m,n in best_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * ratio:
                matches.append(m)
    fim = time.time_ns()
    
    return matches, (fim - inicio) * 1e-9 

def stich(queryImg_path, trainImg_path, feature_extractor, feature_matching):
    logging.debug('Baixando imagem original...')
    logging.debug(f'Caminho: {queryImg_path}')
    queryImg = cv2.imread(queryImg_path, 1)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)

    logging.debug('Baixando imagem a ser costurada...')
    logging.debug(f'Caminho: {trainImg_path}')
    trainImg = cv2.imread(trainImg_path, 1)
    trainImg_gray = cv2.cvtColor(cv2.imread(trainImg_path, 1), cv2.COLOR_BGR2GRAY)
    
    # Busca os pontos-chave e recursos correspondentes às imagens
    logging.info('Encontrando os pontos-chaves...')
    logging.info(f'feature_extractor: {feature_extractor}')
    kpsA, featuresA, tempoA = detectAndDescribe(trainImg_gray, feature_extractor)
    kpsB, featuresB, tempoB = detectAndDescribe(queryImg_gray, feature_extractor)

    dados['key_points'].append([len(kpsA), len(kpsB)])
    dados['feature_extract_time'].append([np.round(tempoA, 4), np.round(tempoB, 4)])

    # Vamos encontrar os ponto chaves correspondentes nessas duas imagens
    logging.info('Encontrando os pares de pontos-chaves correspondentes...')
    logging.info(f'feature_matching: {feature_matching}')
    matches, tempo = matchKeyPoints(featuresA, featuresB, feature_matching, feature_extractor)

    dados['matches'].append(len(matches))
    dados['feature_matching_time'].append(np.round(tempo, 4))

    ## Matriz de Homografia
    # Coordenadas do pontos-chave
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    # Índices dos pontos-chaves no descriptor
    ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

    # DOC: Finds a perspective transformation between two planes. 
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4)

    # Por fim, vamos gerar o resultado final
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    _x = np.nonzero(result.sum(0).sum(-1) == 0)[0][0]
    _y = np.nonzero(result.sum(1).sum(-1) == 0)[0][0]

    return result[:_y,:_x]

lista = [
    {
        "queryImg": 'images/FAD I/imagem_044.png',
        "trainImg": 'images/FAD I/imagem_043.png',
        "pasta": 'FAD I (43 e 44)'
    },
    {
        "queryImg": 'images/FAD I/imagem_053.png',
        "trainImg": 'images/FAD I/imagem_052.png',
        "pasta": 'FAD I (52 e 53)'
    },
    {
        "queryImg": 'images/FAD II/imagem_013.png',
        "trainImg": 'images/FAD II/imagem_012.png',
        "pasta": 'FAD II (12 e 13)'
    },
    {
        "queryImg": 'images/FAD II/imagem_033.png',
        "trainImg": 'images/FAD II/imagem_032.png',
        "pasta": 'FAD II (32 e 33)'
    },
    {
        "queryImg": 'images/FAD II/imagem_040.png',
        "trainImg": 'images/FAD II/imagem_039.png',
        "pasta": 'FAD II (39 e 40)'
    },
    {
        "queryImg": 'images/FE PL/imagem_048.png',
        "trainImg": 'images/FE PL/imagem_047.png',
        "pasta": 'FE PL (47 e 48)'
    },
    {
        "queryImg": 'images/FE PL/imagem_065.png',
        "trainImg": 'images/FE PL/imagem_064.png',
        "pasta": 'FE PL (64 e 65)'
    },
    {
        "queryImg": 'images/FE PL/imagem_078.png',
        "trainImg": 'images/FE PL/imagem_077.png',
        "pasta": 'FE PL (77 e 78)'
    },
]

dados = {
    "extractor": [],
    "key_points": [],
    "feature_extract_time": [],
    "matching": [],
    "matches": [],
    "feature_matching_time": [],
    "images": [],
    "error": []
}

for imagens in lista:
    for extractor in ["brisk", "orb", "sift", "surf"]:
        for matching in ["knn", "bf"]:

            dados['images'].append([imagens["queryImg"], imagens["trainImg"]])
            dados['extractor'].append(extractor)
            dados['matching'].append(matching)
            dados['error'].append(False)

            try:
                nova_imagem = stich(imagens["queryImg"], imagens["trainImg"], extractor, matching)
        
                logging.info(f'Salvando imagem em outputs/{imagens["pasta"]}/{extractor}_{matching}.png')
                cv2.imwrite(f'outputs/{imagens["pasta"]}/{extractor}_{matching}.png', nova_imagem)

            except Exception as erro:
                logging.error(erro)
                dados['error'][-1] = True

pd.DataFrame(dados).to_csv('dados.csv', index=False)