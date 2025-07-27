from flask import Flask, render_template, jsonify
import threading
import cv2

app = Flask(__name__)

camera_running = False
camera_thread = None

def camera_loop():
    global camera_running
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("📷 Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/toggle-camera', methods=['POST'])
def toggle_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.start()
        return jsonify({'status': 'started'})
    else:
        camera_running = False
        return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)
