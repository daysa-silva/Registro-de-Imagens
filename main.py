import argparse
import logging
import cv2
import numpy as np
from funcoes import detectAndDescribe, matchKeyPoints

def stich(queryImg_path, trainImg_path, feature_extractor, feature_matching):
    logging.info('Baixando imagem original...')
    logging.debug(f'Caminho: {queryImg_path}')
    queryImg = cv2.imread(queryImg_path, 1)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)

    logging.info('Baixando imagem a ser costurada...')
    logging.debug(f'Caminho: {trainImg_path}')
    trainImg = cv2.imread(trainImg_path, 1)
    trainImg_gray = cv2.cvtColor(cv2.imread(trainImg_path, 1), cv2.COLOR_BGR2GRAY)
    
    # Busca os pontos-chave e recursos correspondentes às imagens
    logging.info('Encontrando os pontos-chaves...')
    logging.debug(f'feature_extractor: {feature_extractor}')
    kpsA, featuresA = detectAndDescribe(trainImg_gray, feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, feature_extractor)

    # Vamos encontrar os ponto chaves correspondentes nessas duas imagens
    logging.info('Encontrando os pares de pontos-chaves correspondentes...')
    logging.debug(f'feature_matching: {feature_matching}')
    matches = matchKeyPoints(featuresA, featuresB, feature_matching, feature_extractor)


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Costurando 2 imagens',
        description='Dada duas imagens, o programa tenta juntar as duas de acordo com os métodos passados',
        #epilog='Text at the bottom of help'
    )

    parser.add_argument('queryImg', help='Imagem original') 
    parser.add_argument('trainImg', help='Imagem a ser costurada na imagem original')
    parser.add_argument('-o', '--output', default='outputs', help='Pasta onde serão salvos os resultados finais') 
    parser.add_argument('-e', '--extractor', default='orb', help='Método para extrair pontos-chaves. Valores aceitos: brief, orb, sift e surf')
    parser.add_argument('-m', '--matching', default='bf', help='Método para encontrar pares de pontos-chaves correspondentes entre as duas imagens. Valores Aceitos: knn e bf') 
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    nova_imagem = stich(args.queryImg, args.trainImg, args.extractor, args.matching)

    cv2.imwrite(f'{args.output}/{args.extractor}_{args.matching}.png', nova_imagem)