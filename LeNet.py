from tensorflow.keras.models import Sequential  # from keras.models import Sequential 這是1.X.X版本的tensorflow因為keras未整合 所以寫法不同
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class LeNet():
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        first_conv2D = Conv2D(20, (5, 5), padding="same", input_shape=inputShape)
        model.add(first_conv2D)
        first_activation = Activation("relu")
        model.add(first_activation)
        first_pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(first_pooling)

        second_conv2D = Conv2D(50, (5, 5), padding="same")
        model.add(second_conv2D)
        second_activation = Activation("relu")
        model.add(second_activation)
        second_pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        model.add(second_pooling)

        flatten = Flatten()
        model.add(flatten)

        first_dense = Dense(500)
        model.add(first_dense)

        third_activation=Activation("relu")
        model.add(third_activation)

        second_dense=Dense(classes)
        model.add(second_dense)

        fourth_activation=Activation("softmax")
        model.add(fourth_activation)

        return model