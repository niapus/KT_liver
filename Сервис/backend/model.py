from keras import backend as K
K.set_image_data_format('channels_first')
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import io
import os
import uuid
from medpy.io import load
import tempfile

file_name = None


def dice_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    union = K.sum(y_true_f + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

def unet_1(img_channels, image_rows, image_cols, neurons=16):
    inputs = Input((img_channels, image_rows, image_cols))
    conv1 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(neurons * 16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(neurons * 16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Dropout(0.5)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    return model

def get_model(w_size):
    model_1 = unet_1(1, w_size, w_size, 8)

    model_path = r'unet_r_ver28.h5'
    model_1.load_weights(model_path)
    model_1.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    return model_1

def generate_uuid_tiff():
    id = uuid.uuid4().hex[:8]
    return id

# Создаем генератор для нормализации изображения
idg_inference_data = ImageDataGenerator(
    samplewise_center=True,  # Центрирование изображения
    samplewise_std_normalization=True  # Нормализация изображения
)

def load_dicom_image(file_bytes):
    """Загружает DICOM изображение из байтов и извлекает пиксельные данные с использованием medpy."""
    # Создаем временный файл в памяти для хранения байтов
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_bytes)  # Записываем байты в файл
        temp_file_path = temp_file.name  # Получаем путь к временно созданному файлу
    
    # Загружаем изображение из временного файла с использованием medpy
    img, header = load(temp_file_path)
    
    # Удаляем временный файл после загрузки
    os.remove(temp_file_path)
    
    return img

def buffer_single_img(file_bytes, is_dicom):
    """Обрабатывает одно изображение (если это DICOM, то конвертирует в TIFF)."""
    global file_name

    if is_dicom:
        img = load_dicom_image(file_bytes)  # Загружаем изображение из байтового потока
        pil = Image.fromarray(img.squeeze())  # Преобразуем массив в изображение
        pil = pil.transpose(Image.FLIP_TOP_BOTTOM)  # Переворачиваем изображение
        pil = pil.rotate(-90, expand=True)  # Поворачиваем изображение
    else:
        pil = Image.open(io.BytesIO(file_bytes))  # Загружаем изображение из байтового потока
        
    file_name = generate_uuid_tiff()
    fname = "predictions/" + file_name + ".tiff"
    pil.save(fname)  # Сохраняем изображение в файл
    return fname

def load_and_preprocess_image(file_path):
    """Загружает изображение и подготавливает его для подачи в модель (конвертируем в NumPy массив)."""
    img = image.load_img(file_path, target_size=(512, 512), color_mode='grayscale')  # или другой размер
    img_array = image.img_to_array(img)  # Преобразуем изображение в массив
    print(f"Размер Tiff: {img_array.shape}")
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    
    img_array = idg_inference_data.standardize(img_array)
    
    return img_array

def handle_dicom(dicom, is_dicom):
    tiff_file = buffer_single_img(dicom, is_dicom)  # Конвертируем в TIFF
    
    img = Image.open(tiff_file)
    print(f"Пиксели tiff файла max= {np.max(img)}, min= {np.min(img)}")
    
    image_data = load_and_preprocess_image(tiff_file)
    
    print(f"Размер итогового изображения для модели: {image_data.shape}")
    print(f"Его пиксели max= {np.max(image_data)}, min= {np.min(image_data)}")
    
    return image_data

def ndarray_to_image(ndarray):
    ndarray = ndarray.squeeze()  # Убираем размерности 1
    ndarray = (ndarray * 255).astype(np.uint8)  # Приводим к диапазону 0-255

    # Создаем массив RGB с черным фоном
    height, width = ndarray.shape
    colored_array = np.zeros((height, width, 3), dtype=np.uint8)  # Черный фон

    # Добавляем желтый цвет для предикта
    mask = ndarray > 0  # Условие для значений предикта
    colored_array[mask] = [255, 255, 0]  # Желтый цвет (RGB)

    # Преобразуем массив в изображение
    image = Image.fromarray(colored_array)

    # Сохраняем изображение в байтовый поток
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)

    return img_io