import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Input, Convolution2D, BatchNormalization
from keras.initializers import RandomNormal
from keras import optimizers
from keras import callbacks
from keras.callbacks import CSVLogger
import h5py
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio

# 显式设置GPU显存按需增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

def dssim(y_true, y_pred):
    max_val = tf.reduce_max(y_pred)
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=max_val)
    return 1 - ssim_value   # DSSIM 的范围在 [0, 1] 之间

#损失函数 α L1 + （1−α）DSSIM
def custom_loss(y_true, y_pred):
    alpha = 0.8
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    dssim_loss = dssim(y_true, y_pred)
    return alpha * l1_loss + (1 - alpha) * dssim_loss

#训练类
class DnCnn_Class_Train_G:
    #初始化
    def __init__(self):
        self.destFolderName ='./Results/'
        print('Constructor Called')
        self.IMAGE_WIDTH = 50
        self.IMAGE_HEIGHT = 50
        self.CHANNELS = 1
        self.N_SAMPLES = 414720
        self.N_TRAIN_SAMPLES = 384000
        self.N_EVALUATE_SAMPLES = 30720
        self.N_LAYERS = 20
        self.Filters = 64
        self.X_TRAIN = np.zeros((self.N_TRAIN_SAMPLES, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        self.Y_TRAIN = np.zeros((self.N_TRAIN_SAMPLES, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        self.X_EVALUATE = np.zeros((self.N_EVALUATE_SAMPLES, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        self.Y_EVALUATE = np.zeros((self.N_EVALUATE_SAMPLES, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))

        path = 'GDnCNNdata/'
        #加载训练集
        print('train data loading : start')
        with h5py.File(path + 'inputData.mat', 'r') as f:
            xtdata = f['inputData'][()]
            xtdata = np.transpose(xtdata, (2, 0, 1))
            xtdata = np.expand_dims(xtdata, axis=-1)
            self.X_TRAIN[:, :, :, :] = xtdata
        with h5py.File(path + 'labels.mat', 'r') as f:
            ytdata = f['labels'][()]
            ytdata = np.transpose(ytdata, (2, 0, 1))
            ytdata = np.expand_dims(ytdata, axis=-1)
            self.Y_TRAIN[:, :, :, :] = ytdata
        print('train data loading : end')
        #加载验证集
        print('validation data loading : start')
        with h5py.File(path + 'inputDataVal.mat', 'r') as f:
            xedata = f['inputDataVal'][()]
            xedata = np.transpose(xedata, (2, 0, 1))
            xedata = np.expand_dims(xedata, axis=-1)
            self.X_EVALUATE[:, :, :, :] = xedata
        with h5py.File(path + 'labelsVal.mat', 'r') as f:
            yedata = f['labelsVal'][()]
            yedata = np.transpose(yedata, (2, 0, 1))
            yedata = np.expand_dims(yedata, axis=-1)
            self.Y_EVALUATE[:, :, :, :] = yedata
        print('validation data loading : end')

    #构建DnCNN模型结构
    def ModelMaker(self, optim):
        self.myModel = Sequential()
        firstLayer = Convolution2D(filters=self.Filters, kernel_size=(3, 3), strides=(1, 1),
                                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same',
                                   input_shape=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.CHANNELS), use_bias=True,
                                   bias_initializer='zeros')
        self.myModel.add(firstLayer)
        self.myModel.add(Activation('relu'))
        for i in range(self.N_LAYERS - 2):
            Clayer = Convolution2D(filters=self.Filters, kernel_size=(3, 3), strides=(1, 1),
                                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same',
                                   use_bias=True, bias_initializer='zeros')
            self.myModel.add(Clayer)
            Blayer = BatchNormalization(axis=-1, epsilon=1e-3)
            self.myModel.add(Blayer)
            self.myModel.add(Activation('relu'))
        lastLayer = Convolution2D(filters=self.CHANNELS, kernel_size=(3, 3), strides=(1, 1),
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same',
                                  use_bias=True, bias_initializer='zeros')
        self.myModel.add(lastLayer)
        self.myModel.compile(loss=custom_loss, metrics=['mae'], optimizer=optim)
        print("Model Created")
        self.myModel.summary()

    # 复制原模型
    def loadPrevModel(self, modelFileToLoad, optim):
        self.ModelMaker(optim)
        self.savedModel = keras.models.load_model(modelFileToLoad, custom_objects={'dssim': dssim, 'custom_loss': custom_loss})
        self.savedModel.summary()
        self.myModel.set_weights(self.savedModel.get_weights());
        self.myModel.summary()

    # 训练模型
    def trainModelAndSaveBest(self, BATCH_SIZE, EPOCHS, modelFileToSave, logFileToSave, initial_epoch=0):
        # 设置回调函数
        csv_logger = CSVLogger(logFileToSave, append=True)
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=modelFileToSave,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto'
        )
        # 训练模型
        trainHistory = self.myModel.fit(
            x=self.X_TRAIN,
            y=self.Y_TRAIN,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            verbose=1,
            callbacks=[csv_logger, model_checkpoint],
            validation_data=(self.X_EVALUATE, self.Y_EVALUATE)
        )
        return trainHistory

    # 接着之前的轮次继续训练
    def get_current_epoch(self, log_file):
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                return len(f.readlines()) - 1
        return 0

    # 可视化训练结果
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 6))
        # 绘制损失函数曲线
        plt.subplot(2, 1, 1)
        if 'loss' in history.history:
            plt.plot(history.history['loss'], 'b-', linewidth=2, label='训练损失')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], 'r-', linewidth=2, label='验证损失')
        plt.title('损失函数变化曲线', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12)
        # 绘制MAE曲线
        plt.subplot(2, 1, 2)
        if 'mae' in history.history:
            plt.plot(history.history['mae'], 'g-', label='训练MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], 'm-', label='验证MAE')
        plt.title('MAE指标变化', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12)

        plt.tight_layout()
        output_path = os.path.join(self.destFolderName, 'training_history.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存至: {output_path}")
        plt.close()

    # 可视化去噪效果对比
    def visualize_denoising_results(self, num_samples=5):
        # 随机选择样本
        num_total = min(len(self.X_EVALUATE), num_samples)
        indices = np.random.choice(len(self.X_EVALUATE), num_total, replace=False)
        plt.figure(figsize=(15, 5 * num_total))
        for i, idx in enumerate(indices):
            noisy = self.X_EVALUATE[idx]
            residual = self.Y_EVALUATE[idx]
            pred_residual = self.myModel.predict(noisy[np.newaxis, ...])[0]
            clean = noisy - residual
            pred_clean = noisy - pred_residual
            noisy = np.clip(noisy, 0, 1)
            clean = np.clip(clean, 0, 1)
            pred_clean = np.clip(pred_clean, 0, 1)

            plt.subplot(num_total, 3, 3 * i + 1)
            plt.imshow(noisy.squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Noisy {idx}")

            plt.subplot(num_total, 3, 3 * i + 2)
            plt.imshow(pred_clean.squeeze(), cmap='gray', vmin=0, vmax=1)
            psnr = peak_signal_noise_ratio(clean, pred_clean, data_range=1)
            plt.title(f"Denoised\nPSNR:{psnr:.2f}dB")

            plt.subplot(num_total, 3, 3 * i + 3)
            plt.imshow(clean.squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title("Ground Truth")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.destFolderName, 'denoising_comparison.png'))
        plt.close()

    # 重新配置训练
    def reCompileModel(self, optim):
        self.myModel.compile(loss=custom_loss, metrics=['mae'], optimizer=optim)

DnCNN = DnCnn_Class_Train_G();
myOptimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,  amsgrad=False)
model_file = 'DnCNN_G_V1.h5'
log_file = 'DnCNN_G_V1.log'
if os.path.exists(model_file):
    print("发现已有模型，准备继续训练...")
    DnCNN.loadPrevModel(model_file, myOptimizer)
    initial_epoch = DnCNN.get_current_epoch(log_file)
    print(f"将从第 {initial_epoch} 个epoch开始继续训练")
else:
    print("未发现已有模型，创建新模型...")
    DnCNN.ModelMaker(myOptimizer)
    initial_epoch = 0
# 开始/继续训练
myModelHistory = DnCNN.trainModelAndSaveBest(
    BATCH_SIZE=100,
    EPOCHS=50,
    modelFileToSave=model_file,
    logFileToSave=log_file,
    initial_epoch=initial_epoch
)
DnCNN.plot_training_history(myModelHistory)
DnCNN.visualize_denoising_results(num_samples=10)