import cv2
import os
from monitor import FaceHatMonitor
from tqdm import tqdm


def test_by_image_path(image_path):
    model = FaceHatMonitor()
    model.fit()
    print('训练完成...')
    face, hat = model.predict(image_path, True)
    print('身份:%s，安全帽:%s' % (face, '有' if hat == 1 else '无'))


def test_by_camera():
    model = FaceHatMonitor()
    model.fit()
    print('训练完成...')
    video_capture = cv2.VideoCapture(0)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        face, hat = model.predict(frame)
        print('身份:%s，安全帽:%s' % (face, '有' if hat == 1 else '无'))


def test_by_directory(path):
    model = FaceHatMonitor()
    model.fit()
    print('训练完成...')
    face_error = 0
    hat_error = 0
    for dir_name in os.listdir(path):
        if not model._file_filter(dir_name):
            continue
        face_label = dir_name.strip('+安全帽')
        hat_label = ('安全帽' in dir_name) * 1
        dir_path = os.path.join(path, dir_name)
        files = list(filter(model._file_filter, os.listdir(dir_path)))
        for file_name in tqdm(files):
            image_path = os.path.join(dir_path, file_name)
            face_result, hat_result = model.predict(image_path, True,
                                                    plot_to_dir=os.path.join('./result', dir_name),
                                                    plot_to_name=file_name)
            if face_result != face_label:
                print('%s人脸预测错误' % image_path)
                face_error += 1
            if hat_result != hat_label:
                print('%s帽子预测错误' % image_path)
                hat_error += 1
    print('人脸错误%d,帽子错误%d' % (face_error, hat_error))


if __name__ == '__main__':
    # 测试扩增数据
    test_by_directory('./augmentation')

    # 测试训练数据
    # test_by_directory('./data')

    # 测试单张图片
    # test_by_image_path('./augmentation/A/img_0_0_1330.JPG')

    # 测试摄像头
    # test_by_camera()
