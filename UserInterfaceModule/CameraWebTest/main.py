from flask import Flask, render_template, Response
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), 'format': 'BGR888'}))
#picam2.set_controls({"AwbMode": 1})
picam2.start()

time.sleep(2)

def generate_frames():
     frame = picam2.capture_array() # good performance bad color

     # Manual channel adjustment (example values — tune as needed)
     #frame[:, :, 0] = cv2.multiply(frame[:, :, 0], 0.8)  # Blue
     #frame[:, :, 1] = cv2.multiply(frame[:, :, 1], 1.2)  # Green
     #frame[:, :, 2] = cv2.multiply(frame[:, :, 2], 0.6)  # Red

     # swap channels
     #r = frame[:, :, 0]
     #g = frame[:, :, 1]
     #b = frame[:, :, 2]

     #frame[:, :, 0] = b
     #frame[:, :, 2] = r

     ret, buffer = cv2.imencode('.jpg', frame)
     if ret:
         frame_bytes = buffer.tobytes()
         yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#def generate_frames():
    # frame = picam2.capture_array()
#    frame = cv2.imread('D://Desktop//prueba.jpeg')
#    ret, buffer = cv2.imencode('.jpg', frame)
#    frame_bytes = buffer.tobytes()
#    yield (b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
