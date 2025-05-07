import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, BatchNormalization
from keras.initializers import RandomNormal
from keras import optimizers
import matplotlib.pyplot as plt
import imageio
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

def dssim(y_true, y_pred):
    max_val = tf.reduce_max(y_pred)
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=max_val)
    return 1 - ssim_value

def custom_loss(y_true, y_pred):
    alpha = 0.8
    l1_loss = keras.losses.mean_absolute_error(y_true, y_pred)
    dssim_loss = dssim(y_true, y_pred)
    return alpha * l1_loss + (1 - alpha) * dssim_loss

class GDnCNNTester:
    def __init__(self, width, height, colorChannels, destFolderName):
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height
        self.CHANNELS = colorChannels
        self.destFolderName = destFolderName
        self.N_LAYERS = 20
        self.Filters = 64
        # 创建结果目录
        os.makedirs(self.destFolderName, exist_ok=True)
        os.makedirs(os.path.join(self.destFolderName, 'denoised'), exist_ok=True)
        os.makedirs(os.path.join(self.destFolderName, 'noisy'), exist_ok=True)
        os.makedirs(os.path.join(self.destFolderName, 'clean'), exist_ok=True)

    def load_model(self, modelFileToLoad):
        self.model = keras.models.load_model(
            modelFileToLoad,
            custom_objects={'dssim': dssim, 'custom_loss': custom_loss}
        )
        print("Model loaded successfully")
        self.model.summary()

    def evaluate_and_save(self, X_test, indexStart=1):
        # 加载干净图像
        clean_images = []
        for i in range(len(X_test)):
            idx = indexStart + i
            clean_path = os.path.join(clean_data_dir, f"original_{idx:02d}.jpg")
            try:
                clean_img = imageio.v2.imread(clean_path).astype(np.float32) / 255.0
                print(
                    f"Loaded clean image {clean_path} - dtype: {clean_img.dtype}, range: [{clean_img.min()}, {clean_img.max()}]")
                clean_images.append(clean_img)
            except Exception as e:
                print(f"Error loading {clean_path}: {str(e)}")
                clean_images.append(np.zeros_like(X_test[0]))
        clean_images = np.array(clean_images)
        # 预测去噪结果
        denoised = self.model.predict(X_test, batch_size=1, verbose=1)
        denoised = X_test - denoised
        # 计算并保存指标
        metrics = []
        for i in range(len(X_test)):
            index = i + indexStart
            filename = f"{index:04d}"
            self._save_image(X_test[i], 'noisy', filename)
            self._save_image(denoised[i], 'denoised', filename)
            self._save_image(clean_images[i], 'clean', filename)
            # 计算指标
            denoised_i = denoised[i].squeeze()
            clean_image_i = clean_images[i].squeeze()
            if clean_image_i.shape != denoised_i.shape:
                min_shape = (
                    min(clean_image_i.shape[0], denoised_i.shape[0]),
                    min(clean_image_i.shape[1], denoised_i.shape[1])
                )
                denoised_i = denoised_i[:min_shape[0], :min_shape[1]]
                clean_image_i = clean_image_i[:min_shape[0], :min_shape[1]]
            print(clean_image_i.shape, denoised_i.shape)
            if clean_images is not None:
                psnr_val = peak_signal_noise_ratio(
                    clean_image_i, denoised_i, data_range=1.0
                )
                ssim_val = ssim(
                    clean_image_i, denoised_i,
                    data_range=1.0, multichannel=False
                )
                metrics.append((filename, psnr_val, ssim_val))
        # 保存指标到文件
        if metrics:
            with open(os.path.join(self.destFolderName, 'metrics.csv'), 'w') as f:
                f.write("filename,PSNR,SSIM\n")
                for m in metrics:
                    f.write(f"{m[0]},{m[1]:.4f},{m[2]:.4f}\n")
            avg_psnr = np.mean([m[1] for m in metrics])
            avg_ssim = np.mean([m[2] for m in metrics])
            print(f"\nAverage PSNR: {avg_psnr:.4f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
        # 可视化部分结果
        self._visualize_results(X_test[:3], denoised[:3],
                                clean_images[:3] if clean_images is not None else None)

    def _save_image(self, image, subfolder, filename):
        # 确保输入是数值类型
        if isinstance(image, (str, bytes)):
            raise ValueError(f"Image data must be numeric, got {type(image)}")
        # 强制转换为float类型
        image = np.array(image, dtype=np.float32)
        # 处理可能的单通道灰度图（H,W）->（H,W,1）
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        # 裁剪并保存
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8).squeeze()  # 移除单通道维度
        path = os.path.join(self.destFolderName, subfolder, f"{filename}.png")
        imageio.imsave(path, image)

    def _visualize_results(self, noisy, denoised, clean=None):
        plt.figure(figsize=(15, 5 * len(noisy)))
        num_samples = len(noisy)
        for i in range(num_samples):
            # 显示噪声图像
            plt.subplot(3, num_samples, i + 1)
            plt.imshow(noisy[i].squeeze(), cmap='gray')
            plt.title("Noisy")
            plt.axis('off')
            # 显示去噪结果
            plt.subplot(3, num_samples, num_samples + i + 1)
            plt.imshow(denoised[i].squeeze(), cmap='gray')
            plt.title("Denoised")
            plt.axis('off')
            # 显示干净图像
            if clean is not None:
                plt.subplot(3, num_samples, 2 * num_samples + i + 1)
                plt.imshow(clean[i].squeeze(), cmap='gray')
                plt.title("Clean")
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.destFolderName, 'visualization.png'))
        plt.close()

if __name__ == "__main__":
    base_dir = './Results/HDnCNN/'
    test_data_dir = os.path.join(base_dir, 'testdata/')
    clean_data_dir = os.path.join(base_dir, 'Original/')
    output_dir = os.path.join(base_dir, 'predicted/')
    model_path = 'DnCNN_H_V1.h5'
    # 初始化测试器
    tester = GDnCNNTester(width=50, height=50, colorChannels=1, destFolderName=output_dir)
    tester.load_model(model_path)
    # 加载测试数据
    test_files = [f for f in os.listdir(test_data_dir) if f.startswith('testDataCollective')]
    test_files.sort()

    index_start = 1
    for test_file in test_files:
        # 加载.mat文件
        data = sio.loadmat(os.path.join(test_data_dir, test_file))
        X_test = data['testData']
        # 调整数据形状 (N,H,W,C)
        if X_test.ndim == 3:
            X_test = np.expand_dims(X_test, axis=-1)
        elif X_test.shape[-1] != 1:
            X_test = np.transpose(X_test, (0, 2, 3, 1))  # 假设原始是(N,C,H,W)
        # 归一化到[0,1]范围
        X_test = X_test.astype(np.float32)
        # 评估并保存结果
        tester.evaluate_and_save(X_test, indexStart=index_start)
        index_start += len(X_test)