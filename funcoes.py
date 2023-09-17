import cv2
import numpy as np
import logging
import time

def readImage(filename, scale):
    img = cv2.imread(filename, 0)
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))

    if img is None:
        logging.error('Invalid image:' + filename)
        return None
    else:
        logging.info('Image successfully read...')
        return img
    
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
    
    logging.info('Total de pontos-chaves: %d', len(kps))
    logging.debug('Extração levou %.4f segundos', (fim - inicio) * 1e-9)

    return (kps, features)

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

    logging.info('Matches: %d', len(matches))
    logging.debug('Matching levou %.4f segundos', (fim - inicio) * 1e-9 )
    
    return matches
