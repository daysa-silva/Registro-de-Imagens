import argparse
import logging
import os
import cv2
import numpy as np
from src.stich import detectAndDescribe, matchKeyPoints, findHomographyMatrix
from src.utils import readImage, readImageColor

def stitch(queryImg_path, trainImg_path, feature_extractor, feature_matching):
    logging.info('Baixando imagem original...')
    queryImg = readImageColor(queryImg_path, 1)
    queryImg_gray = readImage(queryImg_path, 1)

    logging.info('Baixando imagem a ser costurada...')
    trainImg = readImageColor(trainImg_path, 1)
    trainImg_gray = readImageColor(trainImg_path, 1)
    
    # Busca os pontos-chave e recursos correspondentes às imagens
    logging.info('Encontrando os pontos-chaves...')
    logging.info('feature_extractor: %s', feature_extractor)
    kpsA, featuresA, _ = detectAndDescribe(trainImg_gray, feature_extractor)
    kpsB, featuresB, _ = detectAndDescribe(queryImg_gray, feature_extractor)

    # Vamos encontrar os ponto chaves correspondentes nessas duas imagens
    logging.info('Encontrando os pares de pontos-chaves correspondentes...')
    logging.info('feature_matching: %s', feature_matching)
    matches, _ = matchKeyPoints(featuresA, featuresB, feature_matching, feature_extractor)

    ## Matriz de Homografia
    H, _ = findHomographyMatrix(kpsA, kpsB, matches)

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
    parser.add_argument('-e', '--extractor', default='orb', help='Método para extrair pontos-chaves. Valores aceitos: brisk, orb, sift e surf')
    parser.add_argument('-m', '--matching', default='bf', help='Método para encontrar pares de pontos-chaves correspondentes entre as duas imagens. Valores Aceitos: knn e bf') 
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    nova_imagem = stitch(args.queryImg, args.trainImg, args.extractor, args.matching)

    logging.info(f'Imagens costuradas. Nova imagem de dimensões {nova_imagem.shape}')
    if not os.path.isdir(args.output):
        logging.warning('Diretório %s não existe. Nova magem não será salva', args.output)
    else:
        logging.info(f'Salvando imagem em {args.output}/{args.extractor}_{args.matching}.png')
        cv2.imwrite(f'{args.output}/{args.extractor}_{args.matching}.png', nova_imagem)