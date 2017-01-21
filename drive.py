# ----------------------------------------------------------------------------------
# Project 3 - Behavioral Cloning - drive.py - Chris Gundling
# ----------------------------------------------------------------------------------
'''
This script when run will drive the vehicle autonomously in the simulator
using the trained model.
'''

import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import matplotlib as mpl
import matplotlib.image as mpimg
from scipy.misc import imread, imresize, imsave
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')

def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    #image = mpimg.imread(BytesIO(base64.b64decode(imgString)))
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    image = np.asarray(image)
    # Crop/Change image size
    crop_image = image[40:140,:,:]
    image_size = (66, 200)
    image = imresize(image, size=image_size)

    transformed_image_array = image[None, :, :, :]
    transformed_image_array = (transformed_image_array.astype(np.float32) - 128.) / 128.

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    args = parser.parse_args()

    image_size = (args.resized_image_width, args.resized_image_height)

    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
