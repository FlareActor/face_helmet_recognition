import numpy as np
import os
import cv2
import face_recognition
from tqdm import tqdm
from scipy.stats import skew
from sklearn.svm import SVC
from .params import *


class FaceHatMonitor(object):
    """人脸匹配+安全帽检测模型"""

    def __init__(self):
        self.known_face_encodings = None  # 身份已知的人脸编码
        self.helmet_estimator = None  # 安全帽检测模型

    def fit(self, nb_compare=1):
        """
        训练模型
        :param nb_compare: 用于人脸对比的图像张数
        :return:
        """
        self.known_face_encodings = self.make_comparable_set(nb_compare)
        # 获得安全帽图像的训练集
        helmet_x, helmet_y = self.make_helmet_train_set()
        # 线性核SVM
        self.helmet_estimator = SVC(kernel='linear')
        self.helmet_estimator.fit(helmet_x, helmet_y)

    def predict(self, image):
        """
        检测一张图像中的人脸身份与是否佩戴安全帽
        :param image: np.ndarray图像数据
        :return:
            face_result:人脸身份
            hat_result:是否佩戴安全帽(0否，1是)
        """
        image = self.load_image_file(image, self.preprocess_raw_image)
        face_bbox = self.extract_face_bbox(image)
        if face_bbox is None:
            # print('没有人脸...')
            return None, None
        # 检测人脸
        face_result = self.compare_face(image, [face_bbox], self.known_face_encodings)
        # # 检测帽子
        helmet_image = self.extract_helmet_image(image, face_bbox)
        # 若人脸处于图像顶端，则检测不出安全帽
        if helmet_image.shape[0] == 0 or helmet_image.shape[1] == 0:
            hat_result = 0
        else:
            x_test = self.extract_helmet_features(helmet_image)
            x_test = np.array(x_test).reshape(1, -1)
            hat_result = self.helmet_estimator.predict(x_test)
        # print('%s,%d' % (face_result, hat_result))
        return face_result, hat_result

    def load_image_file(self, image, preprocess=None):
        """
        加载图像并预处理
        :param image: numpy.ndarray图像数据或者图像绝对路径
        :param preprocess: 预处理函数
        :return: numpy.ndarray图像
        """
        if type(image) == str:
            image = face_recognition.load_image_file(image)
        assert type(image) == np.ndarray
        if preprocess is not None:
            image = preprocess(image)
        return image

    def preprocess_raw_image(self, image):
        """对视频原图像进行预处理"""
        image = cv2.resize(image, (RAW_IMAGE_RESIZE_WIDTH, RAW_IMAGE_RESIZE_HEIGHT))
        return image

    def preprocess_helmet_image(self, image):
        """对安全帽图像进行预处理"""
        image = cv2.resize(image, (HELMET_IMAGE_RESIZE_WIDTH, HELMET_IMAGE_RESIZE_HEIGHT))
        return image

    def make_face_encoding(self, image, locations=None):
        """人脸编码"""
        image = self.load_image_file(image)
        return face_recognition.face_encodings(image, locations)[0]

    def compare_face(self, image, face_bbox=None, known_face_encodings=None):
        """
        人脸比对，识别身份
        :param image:
        :param face_bbox: 已知的人脸边框信息
        :param known_face_encodings: 已知的人脸的编码数据
        :return: 图像中的人脸身份
        """
        if known_face_encodings is None:
            known_face_encodings = self.make_comparable_set()
        x_train, y_train = known_face_encodings
        face_encoding = self.make_face_encoding(image, face_bbox)
        face_distances = face_recognition.face_distance(x_train, face_encoding)
        # 统计距离，返回距离最小的人脸身份
        scoreboard = {}
        for label, score in zip(y_train, face_distances):
            if label not in scoreboard:
                scoreboard[label] = 0
            scoreboard[label] += score
        result = min(scoreboard, key=lambda x: scoreboard[x])
        return result

    def make_comparable_set(self, nb_compare=1):
        """
        构造人脸对比集，对每个人生成（nb_compare*2）张人脸数据，用于新样本比对
        :param nb_compare: 对比数量
        :return:
            X:(nb_samples*nb_feats)的np.ndarray人脸编码数组
            y:包含nb_samples个人脸身份标签的list
        """
        X = []
        y = []
        for dir_name in os.listdir(DATA_PATH):
            if not self._file_filter(dir_name):
                continue
            label = dir_name.strip('+安全帽')
            np.random.seed(SEED)
            dir_path = os.path.join(DATA_PATH, dir_name)
            files = list(filter(self._file_filter, os.listdir(dir_path)))
            chosen_files = np.random.choice(files, size=nb_compare)
            for file_path in (os.path.join(dir_path, f) for f in chosen_files):
                image = self.load_image_file(file_path, self.preprocess_raw_image)
                X.append(self.make_face_encoding(image))
                y.append(label)
        X = np.array(X)
        return X, y

    def extract_face_bbox(self, image):
        """
        提取图像中的人脸边框坐标
        :param image:
        :return: (top, right, bottom, left) order
        """
        image = self.load_image_file(image)
        bbox = face_recognition.face_locations(image)
        if len(bbox) > 0:
            return bbox[0]
        return None

    def extract_helmet_image(self, image, face_bounding_box=None):
        """
        提取安全帽图像
        :param image:
        :param face_bounding_box:
        :return:
        """
        image = self.load_image_file(image)
        if face_bounding_box is None:
            face_bounding_box = self.extract_face_bbox(image)
        if face_bounding_box is None:
            return None
        assert len(face_bounding_box) == 4
        top, right, bottom, left = face_bounding_box
        new_top = 2 * top - bottom
        new_top = new_top if new_top > 0 else 0
        new_bottom = top
        helmet = image[new_top:new_bottom, left:right, :]
        return helmet

    def make_helmet_train_set(self):
        """构造安全帽训练集"""
        cache_path = os.path.join(CACHE_PATH, 'helmet_train_set.npz')
        if os.path.exists(cache_path) & CACHE_FLAG:
            npz = np.load(cache_path)
            X = npz['X']
            y = npz['y']
        else:
            X = []
            y = []
            for dir_name in os.listdir(DATA_PATH):
                if not self._file_filter(dir_name):
                    continue
                label = ('安全帽' in dir_name) * 1
                dir_path = os.path.join(DATA_PATH, dir_name)
                files = list(filter(self._file_filter, os.listdir(dir_path)))
                for file_name in tqdm(files):
                    image_path = os.path.join(dir_path, file_name)
                    image = self.load_image_file(image_path, self.preprocess_raw_image)
                    helmet_image = self.extract_helmet_image(image)
                    X.append(self.extract_helmet_features(helmet_image))
                    y.append(label)
            X = np.array(X)
            cache_path = os.path.join(CACHE_PATH, 'helmet_train_set')
            np.savez(cache_path, X=X, y=y)
        return X, y

    def extract_helmet_features(self, image):
        """提取安全帽图像特征"""
        image = self.preprocess_helmet_image(image)
        R = image[:, :, 0].flatten() / 255.
        G = image[:, :, 1].flatten() / 255.
        B = image[:, :, 2].flatten() / 255.
        return [np.mean(R), np.std(R), skew(R),
                np.mean(G), np.std(G), skew(G),
                np.mean(B), np.std(B), skew(B)]

    def _file_filter(self, name):
        """文件过滤"""
        if name.startswith('.'):
            return False
        return True
