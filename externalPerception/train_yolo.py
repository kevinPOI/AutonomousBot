from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
project = rf.workspace("test-5ev0m").project("robot_tracking_bev_subtraction")
version = project.version(2)
dataset = version.download("yolov8")
model = YOLO("yolov8s.pt")       
results = model.train(data="Robot_tracking_bev_subtraction-2/data.yaml", imgsz=640, batch=8, epochs=20, plots=True)  