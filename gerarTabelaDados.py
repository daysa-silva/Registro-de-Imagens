import logging
import cv2
import numpy as np
import pandas as pd
from src.stich import detectAndDescribe, matchKeyPoints, findHomographyMatrix

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
    dados['feature_extract_time (e-04)'].append([np.round(tempoA*10e4), np.round(tempoB*10e4)])
    dados['extract_mean_time (e-06)'].append([np.round(tempoA *10e06 / len(kpsA)), np.round(tempoB *10e06 / len(kpsB))])

    # Vamos encontrar os ponto chaves correspondentes nessas duas imagens
    logging.info('Encontrando os pares de pontos-chaves correspondentes...')
    logging.info(f'feature_matching: {feature_matching}')
    matches, tempo = matchKeyPoints(featuresA, featuresB, feature_matching, feature_extractor)

    dados['matches'].append(len(matches))
    dados['feature_match_time (e-04)'].append(np.round(tempo*10e4))
    dados['match_mean_time (e-06)'].append(np.round(tempo / len(matches) * 10e6))

    ## Matriz de Homografia
    try:
        H, erroHomografia = findHomographyMatrix(kpsA, kpsB, matches)

        dados["rmse_homography"].append(np.round(erroHomografia, 2))
    except Exception as error:
        dados["rmse_homography"].append(np.NAN)
        raise(error)

    # Por fim, vamos gerar o resultado final
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    _x = np.nonzero(result.sum(0).sum(-1) == 0)[0][0]
    _y = np.nonzero(result.sum(1).sum(-1) == 0)[0][0]

    return result[:_y,:_x]

lista = [
    # {
    #     "queryImg": 'images/reduce/casa2.png',
    #     "trainImg": 'images/reduce/casa1.png',
    #     "pasta": 'casa'
    # },
    {
        "queryImg": 'images/FAD I/imagem_005.png',
        "trainImg": 'images/FAD I/imagem_004.png',
        "pasta": 'FAD I (4 e 5)'
    },
    {
        "queryImg": 'images/FAD I/imagem_035.png',
        "trainImg": 'images/FAD I/imagem_034.png',
        "pasta": 'FAD I (34 e 35)'
    },
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
    "feature_extract_time (e-04)": [],
    "extract_mean_time (e-06)": [],
    "matcher": [],
    "matches": [],
    "feature_match_time (e-04)": [],
    "match_mean_time (e-06)": [],
    "rmse_homography": [],
    "images": [],
    "error": []
}

for imagens in lista:
    for extractor in ["brisk", "orb", "sift", "surf"]:
        for matcher in ["knn", "bf"]:

            dados['images'].append([imagens["queryImg"], imagens["trainImg"]])
            dados['extractor'].append(extractor)
            dados['matcher'].append(matcher)
            dados['error'].append(False)

            try:
                nova_imagem = stich(imagens["queryImg"], imagens["trainImg"], extractor, matcher)
        
                # logging.info(f'Salvando imagem em outputs/{imagens["pasta"]}/{extractor}_{matcher}.png')
                # cv2.imwrite(f'outputs/{imagens["pasta"]}/{extractor}_{matcher}.png', nova_imagem)

            except Exception as erro:
                logging.error(erro)
                dados['error'][-1] = True

pd.DataFrame(dados).to_csv('dados.csv', index=False)