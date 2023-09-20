import cv2
import logging

def readImage(filename, scale):
    img = cv2.imread(filename, 0)
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))

    if img is None:
        logging.error('Caminho da imagem inválido: %s', filename)
        return None
    else:
        logging.debug('Imagem %s baixada com sucesso...', filename)
        return img
    
def readImageColor(filename, scale):
    img = cv2.imread(filename)
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    
    if img is None:
        logging.error('Invalid image: %s', filename)
        return None
    else:
        logging.debug('Imagem %s baixada com sucesso...', filename)
        return img