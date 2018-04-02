import numpy as np
import os
import cv2
import face_recognition
from scipy.stats import skew
from sklearn.svm import SVC
from multiprocessing import Pool
from .params import *


class FaceHatMonitor(object):
    """人脸匹配+安全帽检测模型"""

    def __init__(self):
        self.known_face_encodings = None  # 身份已知的人脸编码
        self.helmet_estimator = None  # 安全帽检测模型
        self._process_pool = Pool()  # multi process

    def fit(self, nb_compare=1):
        """
        训练模型
        :param nb_compare: 用于人脸对比的图像张数
        :return:
        """
        self.known_face_encodings = self._make_comparable_set(nb_compare)
        # 获得安全帽图像的训练集
        helmet_x, helmet_y = self._make_helmet_train_set()
        # 线性核SVM
        self.helmet_estimator = SVC(kernel='rbf', random_state=0)
        self.helmet_estimator.fit(helmet_x, helmet_y)
        self._process_pool.close()

    def predict(self, image, plot=False,
                plot_to_dir=PLOT_PATH, plot_to_name='default.JPG'):
        """
        检测一张图像中的人脸身份与是否佩戴安全帽
        :param plot: 是否绘制人脸框并保存图像
        :param image: np.ndarray图像数据
        :param plot_to_name:文件名
        :param plot_to_dir:文件夹路径
        :return:
            face_result:人脸身份
            hat_result:是否佩戴安全帽(0否，1是)
        """
        image = self._load_image_file(image, self._preprocess_raw_image)
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
            x_test = self._extract_helmet_features(helmet_image)
            x_test = np.array(x_test).reshape(1, -1)
            hat_result = self.helmet_estimator.predict(x_test)
        if plot:
            # 绘制结果并保存
            top, right, bottom, left = face_bbox
            image2draw = image.copy()
            cv2.rectangle(image2draw, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(image2draw, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text2draw = 'face:%s,helmet:%s' % (face_result, 'Yes' if hat_result == 1 else 'No')
            cv2.putText(image2draw, text2draw, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            os.makedirs(plot_to_dir, exist_ok=True)
            cv2.imwrite(os.path.join(plot_to_dir, plot_to_name), image2draw)
        # print('%s,%d' % (face_result, hat_result))
        return face_result, hat_result

    @staticmethod
    def _preprocess_raw_image(image):
        """对视频原图像进行预处理"""
        image = cv2.resize(image, (RAW_IMAGE_RESIZE_WIDTH, RAW_IMAGE_RESIZE_HEIGHT))
        image = image[:, :, ::-1]
        return image

    @staticmethod
    def _preprocess_helmet_image(image):
        """对安全帽图像进行预处理"""
        image = cv2.resize(image, (HELMET_IMAGE_RESIZE_WIDTH, HELMET_IMAGE_RESIZE_HEIGHT))
        return image

    def compare_face(self, image, face_bbox=None, known_face_encodings=None):
        """
        人脸比对，识别身份
        :param image:
        :param face_bbox: 已知的人脸边框信息
        :param known_face_encodings: 已知的人脸的编码数据
        :return: 图像中的人脸身份
        """
        if known_face_encodings is None:
            known_face_encodings = self._make_comparable_set()
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

    def __getstate__(self):
        """控制对象的序列化(pickle and unpickle)
        https://stackoverflow.com/questions/25382455/python-notimplementederror-pool-objects-cannot-be-passed-between-processes
        """
        new_dict = self.__dict__.copy()  # 浅拷贝
        del new_dict['_process_pool']
        return new_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _parallel_func_1(self, image_path):
        image = self._load_image_file(image_path, self._preprocess_raw_image)
        return self.make_face_encoding(image)

    def _parallel_func_2(self, image_path):
        image = self._load_image_file(image_path, self._preprocess_raw_image)
        helmet_image = self.extract_helmet_image(image)
        return self._extract_helmet_features(helmet_image)

    def _make_comparable_set(self, nb_compare=1):
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
                async_result = self._process_pool.apply_async(self._parallel_func_1, (file_path,))
                X.append(async_result)
                y.append(label)
        X = [result.get() for result in X]  # 阻塞等待
        X = np.array(X)
        return X, y

    def _make_helmet_train_set(self):
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
                for file_name in files:
                    image_path = os.path.join(dir_path, file_name)
                    async_result = self._process_pool.apply_async(self._parallel_func_2, (image_path,))
                    X.append(async_result)
                    y.append(label)
            X = [result.get() for result in X]  # 阻塞等待
            X = np.array(X)
            cache_path = os.path.join(CACHE_PATH, 'helmet_train_set')
            np.savez(cache_path, X=X, y=y)
        return X, y

    def extract_helmet_image(self, image, face_bounding_box=None):
        """
        提取安全帽图像
        :param image:
        :param face_bounding_box:
        :return:
        """
        image = self._load_image_file(image)
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

    @classmethod
    def _extract_helmet_features(cls, image):
        """提取安全帽图像特征"""
        image = cls._preprocess_helmet_image(image)
        R = image[:, :, 0].flatten() / 255.
        G = image[:, :, 1].flatten() / 255.
        B = image[:, :, 2].flatten() / 255.
        return [np.mean(R), np.std(R), skew(R),
                np.mean(G), np.std(G), skew(G),
                np.mean(B), np.std(B), skew(B)]

    def make_face_encoding(self, image, locations=None):
        """人脸编码"""
        image = self._load_image_file(image)
        return face_recognition.face_encodings(image, locations)[0]

    def extract_face_bbox(self, image):
        """
        提取图像中的人脸边框坐标
        :param image:
        :return: (top, right, bottom, left) order
        """
        image = self._load_image_file(image)
        bbox = face_recognition.face_locations(image)
        if len(bbox) > 0:
            return bbox[0]
        return None

    @staticmethod
    def _load_image_file(image, preprocess=None):
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

    @staticmethod
    def _file_filter(name):
        """文件过滤"""
        if name.startswith('.'):
            return False
        return True
