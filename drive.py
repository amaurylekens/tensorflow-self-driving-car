import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO
from SDC_CNN import SDC_CNN
import numpy as np
import scipy
#import cv2

# create a Socket.IO server
sio = socketio.Server()

# create cnn
sdc_cnn = SDC_CNN([66,200,3])
i = 1

# event sent by the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        global i
        i += 1
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image_resized = scipy.misc.imresize(image, [66, 200])

        #cv2.imwrite("gif/frame" + str(i) + ".png", image)

        steer = sdc_cnn.predict(image_resized)

        speed_limit = 0
        if speed > 25:
            speed_limit = 10
        else:
            speed_limit = 25

        steer = steer[0,0]
        throttle = 0.25 - steer**2 -(speed/speed_limit)**2

        # response to the simulator with a steer angle and throttle
        send(steer, throttle)
    else:
        # Edge case
        sio.emit('manual', data={}, skip_sid=True)

# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send(0, 0)

# to send steer angle and throttle to the simulator
def send(steer, throttle):
    print("steer : {0}".format(steer*25))
    sio.emit("steer", data={'steering_angle': str(steer), 'throttle': str(throttle)}, skip_sid=True)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# simulator will connect to localhost:4567
if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)