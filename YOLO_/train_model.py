from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = YOLO('yolo11s.pt')

    model.train(
        data='./data.yaml',  # 目标检测数据集配置
        epochs=100,  # 训练 2 个 Epoch
        batch=0.86,  # 每次训练 (16张 或者 GPU占用量 0.8(80%))
        device='0',  # 使用 GPU 0 进行训练
        imgsz= 1024,  # 训练图片大小
        workers=6,  # 使用 2 个 CPU 线程进行数据加载
        cache=False,  # 是否缓存数据（提升训练速度）
        amp=True,  # 使用混合精度训练（提高性能）
        mosaic=True,  # 禁用 Mosaic 数据增强（默认启用）
        project='E:/terraria_project/test',  # 训练结果存放目录
        name='terraria_0326'  # 文件名称
    )