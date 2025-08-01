from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import time
from ultralytics import YOLO
import threading
import queue
import os
import psutil

app = Flask(__name__)

MODEL_PATH = r'C:\Users\lapt1\Downloads\config-file\ppe.pt'
RTSP_URLS = [
    r'rtsp://36.92.47.218:7430/video1',
    r'rtsp://36.92.47.218:7430/video3',
    r'rtsp://36.92.47.218:7430/video4',
    r'rtsp://36.92.47.218:7430/video6',
]

DETECTION_IMG_SIZE = 640
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540

performance_data = [{} for _ in range(len(RTSP_URLS))]
history_data = [[] for _ in range(len(RTSP_URLS))]

total_detections = [0 for _ in range(len(RTSP_URLS))]

class CameraStream:
    def __init__(self, rtsp_url, model, camera_id):
        self.rtsp_url = rtsp_url
        self.model = model
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.cap = None
        self.last_frame_time = time.time()
        self.status = "Loading"

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.cap:
            self.cap.release()

    def _run(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            self.status = "Stream Failed"
            return
        self.status = "Running"

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            try:
                start = time.time()
                results = self.model.predict(source=frame, imgsz=DETECTION_IMG_SIZE, conf=0.5, verbose=False)
                annotated_img = results[0].plot()
                annotated_img = cv2.resize(annotated_img, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
                fps = 1 / (time.time() - self.last_frame_time)
                self.last_frame_time = time.time()

                detected_objects = len(results[0].boxes)
                total_detections[self.camera_id] += detected_objects

                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent

                performance_data[self.camera_id] = {
                    'fps': round(fps, 2),
                    'cpu': cpu,
                    'ram': ram,
                    'detections': detected_objects,
                    'status': self.status,
                    'total_detections': total_detections[self.camera_id]
                }

                history_data[self.camera_id].append(performance_data[self.camera_id])
                if len(history_data[self.camera_id]) > 60:
                    history_data[self.camera_id] = history_data[self.camera_id][-60:]

                if not self.frame_queue.full():
                    self.frame_queue.put(annotated_img)
            except Exception as e:
                self.status = f"Error: {e}"
                continue

def load_yolo_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan di {path}")
    return YOLO(path)

yolo_model = load_yolo_model(MODEL_PATH)
camera_streams = []
for i, url in enumerate(RTSP_URLS):
    stream = CameraStream(url, yolo_model, i)
    stream.start()
    camera_streams.append(stream)

def generate_frames(camera_id):
    stream = camera_streams[camera_id]
    while True:
        if not stream.frame_queue.empty():
            frame = stream.frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template_string(r'''
    <html>
    <head>
        <title>Antares AI Detection</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background-color: #f4f4f4; min-height: 100vh; }
            .header { display: flex; align-items: center; padding: 20px; background: #ffffff; border-bottom: 1px solid #ddd; }
            .header img { height: 40px; margin-right: 12px; }
            .header h2 { margin: 0; font-size: 24px; }
            .camera-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 16px; background: #fff; }
            .camera-box { background: #f9f9f9; padding: 10px; border-radius: 12px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
            .camera-box img { width: 100%; max-height: 180px; object-fit: contain; border-radius: 10px; }
            .charts-container { display: flex; flex-direction: row; gap: 16px; padding: 16px; background: #fff; flex-wrap: nowrap; min-height: 400px; }
            .chart-box {
                flex: 1;
                background: #ffffff;
                padding: 16px;
                border-radius: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                min-height: 340px;
                display: flex;
                flex-direction: column;
            }
            .chart-box canvas {
                flex-grow: 1;
                width: 100% !important;
                height: 260px !important;
                max-height: 260px;
                box-sizing: border-box;
            }
            .summary-container {
                display: flex;
                justify-content: space-around;
                background: #fff;
                padding: 16px;
                border-top: 1px solid #ddd;
            }
            .summary-card {
                padding: 12px 20px;
                background: #f0f0f0;
                border-radius: 10px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.1);
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="camera-container">
        {% for i in range(4) %}
            <div class="camera-box">
                <img src="/video_feed/{{ i }}">
                <p>Status: {{ performance_data[i]['status'] }}</p>
            </div>
        {% endfor %}
        </div>
        
        <div class="summary-container">
            {% for i in range(4) %}
            <div class="summary-card" id="card-{{ i }}">
                <strong>Camera {{ i+1 }}</strong><br>
                Total Detections: {{ performance_data[i]['total_detections'] }}<br>
                Current FPS: {{ performance_data[i]['fps'] }}
            </div>
            {% endfor %}
        </div>

        <div class="charts-container">
            <div class="chart-box">
                <h3>Real-Time Performance</h3>
                <canvas id="barChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>FPS History</h3>
                <canvas id="historyChart"></canvas>
            </div>
        </div>

        <div class="summary-container" style="margin-top: 0;">
            <div class="summary-card" id="avg-load">
                <strong>System Load</strong><br>
                CPU Avg: 0%<br>
                RAM Avg: 0%
            </div>
            <div class="summary-card" id="best-detector">
                <strong>Top Detection</strong><br>
                Camera -<br>
                Total: -
            </div>
            <div class="summary-card" id="lowest-fps">
                <strong>Lowest FPS</strong><br>
                Camera -<br>
                FPS: -
            </div>
        </div>

        <script>
        function getStatusColor(status) {
            if (status.includes("Error") || status.includes("Failed")) return '#f8d7da';
            if (status.includes("Inisialisasi") || status.includes("Loading")) return '#fff3cd';
            if (status.includes("Berjalan") || status.includes("Running")) return '#d4edda';
            return '#e2e3e5';
        }

        const ctx = document.getElementById('barChart').getContext('2d');
        const barChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4'],
                datasets: [
                    { label: 'FPS', data: [0,0,0,0], backgroundColor: 'rgba(144, 190, 183, 0.6)' },
                    { label: 'CPU %', data: [0,0,0,0], backgroundColor: 'rgba(249, 168, 117, 0.6)' },
                    { label: 'RAM %', data: [0,0,0,0], backgroundColor: 'rgba(196, 181, 253, 0.6)' },
                    { label: 'Detections', data: [0,0,0,0], backgroundColor: 'rgba(255, 173, 173, 0.6)' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y}`;
                            }
                        }
                    }
                }
            }
        });

        const hctx = document.getElementById('historyChart').getContext('2d');
        const historyChart = new Chart(hctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 60}, (_, i) => i),
                datasets: [
                    { label: 'Cam 1 FPS', data: [], borderColor: 'rgba(144, 190, 183, 1)', tension: 0.4, fill: false },
                    { label: 'Cam 2 FPS', data: [], borderColor: 'rgba(249, 168, 117, 1)', tension: 0.4, fill: false },
                    { label: 'Cam 3 FPS', data: [], borderColor: 'rgba(196, 181, 253, 1)', tension: 0.4, fill: false },
                    { label: 'Cam 4 FPS', data: [], borderColor: 'rgba(255, 173, 173, 1)', tension: 0.4, fill: false }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { beginAtZero: true },
                    x: { title: { display: true, text: 'Last Seconds' } }
                }
            }
        });

        async function updateChart() {
            const res = await fetch('/performance');
            const data = await res.json();
            for (let i = 0; i < 4; i++) {
                barChart.data.datasets[0].data[i] = data[i].fps || 0;
                barChart.data.datasets[1].data[i] = data[i].cpu || 0;
                barChart.data.datasets[2].data[i] = data[i].ram || 0;
                barChart.data.datasets[3].data[i] = data[i].detections || 0;

                document.querySelector(`#card-${i}`).style.backgroundColor = getStatusColor(data[i].status);
                document.querySelector(`#card-${i}`).title = `Status: ${data[i].status}`;
                document.querySelector(`#card-${i}`).innerHTML = `<strong>Camera ${i+1}</strong><br>Total Detections: ${data[i].total_detections}<br>Current FPS: ${data[i].fps}`;

                document.querySelectorAll('.camera-box')[i].style.backgroundColor = getStatusColor(data[i].status);
                document.querySelectorAll('.camera-box p')[i].textContent = `Status: ${data[i].status}`;
            }
            let totalCPU = 0, totalRAM = 0;
            let topDetections = -1, topCam = '-', lowestFPS = Infinity, lowCam = '-';

            for (let i = 0; i < 4; i++) {
                totalCPU += data[i].cpu;
                totalRAM += data[i].ram;

                if (data[i].total_detections > topDetections) {
                    topDetections = data[i].total_detections;
                    topCam = `Camera ${i + 1}`;
                }

                if (data[i].fps < lowestFPS) {
                    lowestFPS = data[i].fps;
                    lowCam = `Camera ${i + 1}`;
                }
            }

            document.getElementById('avg-load').innerHTML = `<strong>System Load</strong><br>CPU Avg: ${(totalCPU/4).toFixed(1)}%<br>RAM Avg: ${(totalRAM/4).toFixed(1)}%`;
            document.getElementById('best-detector').innerHTML = `<strong>Top Detection</strong><br>${topCam}<br>Total: ${topDetections}`;
            document.getElementById('lowest-fps').innerHTML = `<strong>Lowest FPS</strong><br>${lowCam}<br>FPS: ${lowestFPS.toFixed(2)}`;

            barChart.update();
        }

        async function updateHistory() {
            const res = await fetch('/history');
            const hist = await res.json();
            for (let i = 0; i < 4; i++) {
                historyChart.data.datasets[i].data = hist[i].map(d => d.fps || 0);
            }
            historyChart.update();
        }

        setInterval(() => { updateChart(); updateHistory(); }, 1000);
        </script>
    </body>
    </html>
    ''', performance_data=performance_data)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/performance')
def performance():
    return jsonify(performance_data)

@app.route('/history')
def history():
    return jsonify(history_data)

if __name__ == '__main__':
    app.run(debug=False)
