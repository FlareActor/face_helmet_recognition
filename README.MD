## 项目结构

.

├── README.MD

├── augmentation                          # 扩增数据，当作测试集

├── cache 					       # 缓存文件

│   └── helmet_train_set.npz

├── data

├── result   					# 保存识别后的图片结果

├── data_augmentation.py            # 扩增数据

├── main.py					# 测试入口

└── monitor

        ├── __init__.py

        ├── models.py				# 模型

        └── params.py			# 配置参数

## 参考

https://github.com/ageitgey/face_recognition