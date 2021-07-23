from abc import ABC, abstractmethod
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Model


class Autoencoder(ABC):
    @abstractmethod
    def get_model(self):
        raise (NotImplementedError)


class SimpleAutoencoder(Autoencoder):
    def get_model(self):
        encoder = Input((640, 360, 3))
        conv1 = Conv2D(
            64, kernel_size=(3, 3), padding="same", activation="elu"
        )(encoder)
        conv2 = Conv2D(
            64, kernel_size=(3, 3), padding="same", activation="elu"
        )(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(
            32, kernel_size=(3, 3), padding="same", activation="elu"
        )(pool1)
        conv4 = Conv2D(
            32, kernel_size=(3, 3), padding="same", activation="elu"
        )(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(
            16, kernel_size=(3, 3), padding="same", activation="elu"
        )(pool2)
        conv6 = Conv2D(
            16, kernel_size=(3, 3), padding="same", activation="elu"
        )(conv5)
        encoder_output = MaxPooling2D(pool_size=(2, 2))(conv6)

        dec1 = Conv2D(16, kernel_size=(3, 3), padding="same", activation="elu")(
            encoder_output
        )
        dec2 = Conv2D(16, kernel_size=(3, 3), padding="same", activation="elu")(
            dec1
        )
        up1 = UpSampling2D((2, 2))(dec2)
        dec3 = Conv2D(32, kernel_size=(3, 3), padding="same", activation="elu")(
            up1
        )
        dec4 = Conv2D(32, kernel_size=(3, 3), padding="same", activation="elu")(
            dec3
        )
        up2 = UpSampling2D((2, 2))(dec4)
        dec5 = Conv2D(64, kernel_size=(3, 3), padding="same", activation="elu")(
            up2
        )
        dec6 = Conv2D(64, kernel_size=(3, 3), padding="same", activation="elu")(
            dec5
        )
        up3 = UpSampling2D((2, 2))(dec6)
        decoder_output = Conv2D(
            3, kernel_size=(3, 3), padding="same", activation="sigmoid"
        )(up3)

        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        model = Model(encoder, decoder_output)
        model.compile(optimizer=optimizer, loss="binary_crossentropy")

        return encoder, decoder, model
