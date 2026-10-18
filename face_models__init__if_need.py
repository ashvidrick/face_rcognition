import os

# Получаем абсолютный путь к текущей папке с моделями
_base_dir = os.path.dirname(os.path.abspath(__file__))

def pose_predictor_five_point_model_location():
    return os.path.join(_base_dir, 'models', 'shape_predictor_5_face_landmarks.dat')

def pose_predictor_model_location():
    return os.path.join(_base_dir, 'models', 'shape_predictor_68_face_landmarks.dat')

def face_recognition_model_location():
    return os.path.join(_base_dir, 'models', 'dlib_face_recognition_resnet_model_v1.dat')

def cnn_face_detector_model_location():
    return os.path.join(_base_dir, 'models', 'mmod_human_face_detector.dat')