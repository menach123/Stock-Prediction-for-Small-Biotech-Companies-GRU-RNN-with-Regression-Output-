import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image


def data_to_tensor(df):
    """
    Create a image tensor.
    Input-Dataframe from Yahoo API.   Output- Image Tensor (1, 224, 224, 3) 
    """
    if len(df) == 0:
        return []
    print(df.High.iloc[0])
    fig, ax = plt.subplots(figsize=(10,8))
    #High, low, and close price on plot
    plt.subplot(411)
    df.High.plot(color='g')
    df.Low.plot(color='r')
    df.Close.plot(color='b')
    plt.axis('off')
    #High, low and open price on plot
    plt.subplot(412)
    df.High.plot(color='g')
    df.Low.plot(color='r')
    df.Open.plot(color='b')
    plt.axis('off')
    #Adj Close on plot
    plt.subplot(413)
    df['Adj Close'].plot(color='g')
    plt.axis('off')
    #Volume is plotted.
    plt.subplot(414)
    df.Volume.plot(color='r')
    plt.axis('off')
    #Save plot as an image
    fig.savefig('image.png')
    plt.close(fig)
    img = image.load_img('image.png', target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    #Follow the Original Model Preprocessing
    img_tensor /= 255.
    return img_tensor