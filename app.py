from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import base64

app = Flask(__name__)

cap = None
is_recording = False

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'hi', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def generate_processed_video():
    global cap

    while is_recording:
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global cap, is_recording

    if not is_recording:
        return Response()

    image_data = request.form['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_frame = process_frame(frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    response_image = buffer.tobytes()

    return Response(response=response_image, content_type='image/jpeg')

@app.route('/start', methods=['POST'])
def start():
    global cap, is_recording

    if not is_recording:
        cap = cv2.VideoCapture(0)
        is_recording = True

    return Response()

@app.route('/stop', methods=['POST'])
def stop():
    global cap, is_recording

    if is_recording:
        is_recording = False
        cap.release()

    return Response()

if __name__ == '__main__':
    app.run(debug=True)
