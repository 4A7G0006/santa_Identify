from imutils import paths
import random
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from LeNet import LeNet
from tensorflow.keras.optimizers import Adam  # 優化演算法
import matplotlib.pyplot as plt

print("[INFO] loading images......")
data = []
labels = []
imagePaths = paths.list_images('.\images')
imagePaths = list(imagePaths)
imagePaths = sorted(imagePaths)
random.seed(42)
random.shuffle(imagePaths)  # 打散圖片

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]  # sep判斷系統的'\' '/' 問題  #假如路徑 '/images/santa/000001.jpg' 分割 '/' 取[-2] santa
    label = 1 if label == "santa" else 0
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)  # 資料分類 訓練資料與測試資料 0.75 0.25這樣
# trainY=[0,1,0,1,0,0,0,1]
# =>[[1,0],[0,1],[1,0],[1,0],[1,0]] # ONE HOT encoding
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")  # 圖像翻轉 加強學習

print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)  # 呼叫剛剛所寫的模型
EPOCHS = 25  # 訓練參數設定 #訓練次數25次
INIT_LR = 1e-3  # 訓練率 靠體感
BS = 32  # 每次圖檔抓取32張 疊代ㄧ次32張
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)  # 呼叫優化演算法
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])  # 給定模型的loss function為binary 二分類損失 優化器使用剛剛呼叫的Adam優化 評估指標選用accuracy
# 因為剛剛使用資料增強 所以使用fit訓練 同時避免記憶體不足 如果不使用就使用ㄧ般generator就好
#aug.flow設定資料流 傳入圖像訓練集X 與 訓練標籤 Y  #validation_data傳入測試資料集與標籤  如同前面訓練資料ㄧ般 #steps_per_epoch訓練資料數據除以批次訓練大小就會達到一個新的EPOCH   #EPOCH跑25遍資料 #verbose打印資料訓練log
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)
model.save('./santa_not_santa.model')

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('./plot.png')