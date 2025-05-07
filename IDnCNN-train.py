import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Input, Convolution2D, BatchNormalization
from keras.initializers import RandomNormal
from keras import optimizers
from keras import callbacks
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import h5py
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio

# GPU配置
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
    return 1 - ssim_value

def custom_loss(y_true, y_pred):
    alpha = 0.8
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    dssim_loss = dssim(y_true, y_pred)
    return alpha * l1_loss + (1 - alpha) * dssim_loss

# 数据生成器（解决因数据量太大而无法训练的问题）
class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))
        if self.X is None or self.Y is None:
            raise ValueError("Input data cannot be None")
        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must have the same length")

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.X))
        batch_x = self.X[start_idx:end_idx]
        batch_y = self.Y[start_idx:end_idx]
        if batch_x is None or batch_y is None:
            raise ValueError(f"Batch {idx} contains None values")
        if np.any(np.isnan(batch_x)) or np.any(np.isnan(batch_y)):
            raise ValueError(f"Batch {idx} contains NaN values")
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class DnCnn_Class_Train_I:
    def __init__(self):
        self.destFolderName = './Results/'
        os.makedirs(self.destFolderName, exist_ok=True)
        print('Constructor Called')
        self.IMAGE_WIDTH = 50
        self.IMAGE_HEIGHT = 50
        self.CHANNELS = 1
        self.N_SAMPLES = 414720
        self.N_TRAIN_SAMPLES = 384000
        self.N_EVALUATE_SAMPLES = 30720
        self.N_LAYERS = 20
        self.Filters = 64
        self.X_TRAIN = None
        self.Y_TRAIN = None
        self.X_EVALUATE = None
        self.Y_EVALUATE = None
        self.load_data()

    # 加载数据
    def load_data(self):
        path = 'IDnCNNdata'
        # 加载训练集
        print('Loading training data...')
        try:
            with h5py.File(path + 'inputData.mat', 'r') as f:
                xtdata = np.transpose(f['inputData'][()], (2, 0, 1))
                self.X_TRAIN = np.expand_dims(xtdata[:self.N_TRAIN_SAMPLES], -1).astype('float32')
            with h5py.File(path + 'labels.mat', 'r') as f:
                ytdata = np.transpose(f['labels'][()], (2, 0, 1))
                self.Y_TRAIN = np.expand_dims(ytdata[:self.N_TRAIN_SAMPLES], -1).astype('float32')
            self.X_TRAIN = self.normalize_data(self.X_TRAIN)
            self.Y_TRAIN = self.normalize_data(self.Y_TRAIN)
        except Exception as e:
            raise ValueError(f"Error loading training data: {str(e)}")
        # 加载验证集
        print('Loading validation data...')
        try:
            with h5py.File(path + 'inputDataVal.mat', 'r') as f:
                xedata = np.transpose(f['inputDataVal'][()], (2, 0, 1))
                self.X_EVALUATE = np.expand_dims(xedata[:self.N_EVALUATE_SAMPLES], -1).astype('float32')
            with h5py.File(path + 'labelsVal.mat', 'r') as f:
                yedata = np.transpose(f['labelsVal'][()], (2, 0, 1))
                self.Y_EVALUATE = np.expand_dims(yedata[:self.N_EVALUATE_SAMPLES], -1).astype('float32')
            self.X_EVALUATE = self.normalize_data(self.X_EVALUATE)
            self.Y_EVALUATE = self.normalize_data(self.Y_EVALUATE)
        except Exception as e:
            raise ValueError(f"Error loading validation data: {str(e)}")
        print("Data loaded successfully")
        self.validate_data()

    def normalize_data(self, data):
        if data is None:
            return None
        data_min = np.min(data)
        data_max = np.max(data)
        return (data - data_min) / (data_max - data_min + 1e-8)

    def validate_data(self):
        print("Validating data...")
        datasets = [
            (self.X_TRAIN, 'X_TRAIN'),
            (self.Y_TRAIN, 'Y_TRAIN'),
            (self.X_EVALUATE, 'X_EVALUATE'),
            (self.Y_EVALUATE, 'Y_EVALUATE')
        ]
        for data, name in datasets:
            if data is None:
                raise ValueError(f"{name} is None")
            if np.any(np.isnan(data)):
                raise ValueError(f"{name} contains NaN values")
            if np.any(np.isinf(data)):
                raise ValueError(f"{name} contains Inf values")
            print(f"{name} shape: {data.shape}, dtype: {data.dtype}, min: {np.min(data):.4f}, max: {np.max(data):.4f}")
        print("Data validation passed")

    def ModelMaker(self, optim):
        self.myModel = Sequential()
        firstLayer = Convolution2D(
            filters=self.Filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
            padding='same',
            input_shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS),
            use_bias=True,
            bias_initializer='zeros'
        )
        self.myModel.add(firstLayer)
        self.myModel.add(Activation('relu'))
        for _ in range(self.N_LAYERS - 2):
            self.myModel.add(Convolution2D(
                filters=self.Filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                padding='same',
                use_bias=True,
                bias_initializer='zeros'
            ))
            self.myModel.add(BatchNormalization(axis=-1, epsilon=1e-3))
            self.myModel.add(Activation('relu'))
        lastLayer = Convolution2D(
            filters=self.CHANNELS,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
            padding='same',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.myModel.add(lastLayer)
        self.myModel.compile(
            loss=custom_loss,
            metrics=['mae'],
            optimizer=optim
        )
        print("Model created successfully")
        self.myModel.summary()
    def trainModelAndSaveBest(self, BATCH_SIZE, EPOCHS, modelFileToSave, logFileToSave):
        if self.X_TRAIN is None or self.Y_TRAIN is None:
            raise ValueError("Training data not loaded")
        train_gen = DataGenerator(self.X_TRAIN, self.Y_TRAIN, BATCH_SIZE)
        val_gen = DataGenerator(self.X_EVALUATE, self.Y_EVALUATE, BATCH_SIZE)
        callbacks_list = [
            CSVLogger(logFileToSave),
            ModelCheckpoint(
                filepath=modelFileToSave if modelFileToSave.endswith('.keras') else modelFileToSave + '.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='auto',
                verbose=1
            )
        ]
        # 训练模型
        print("Starting training...")
        try:
            history = self.myModel.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks_list,
                verbose=1
            )
            print("Training completed successfully")
            return history
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

    def loadPrevModel(self, modelFileToLoad, optim):
        self.ModelMaker(optim)
        self.savedModel = keras.models.load_model(modelFileToLoad, custom_objects={'dssim': dssim, 'custom_loss': custom_loss})
        self.savedModel.summary()
        self.myModel.set_weights(self.savedModel.get_weights());
        self.myModel.summary()

    def visualize_denoising_results(self, num_samples=5):
        # 随机选择样本（确保不超出范围）
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
            # 显示
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
        print("Visualization saved")

# 训练流程
if __name__ == "__main__":
    try:
        DnCNN = DnCnn_Class_Train_I()
        myOptimizer = optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False
        )
        DnCNN.ModelMaker(myOptimizer)
        myModelHistory = DnCNN.trainModelAndSaveBest(
            BATCH_SIZE=100,
            EPOCHS=50,
            modelFileToSave='DnCNN_I_V1.keras',
            logFileToSave='DnCNN_I_V1.log'
        )
        DnCNN.visualize_denoising_results(num_samples=10)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise