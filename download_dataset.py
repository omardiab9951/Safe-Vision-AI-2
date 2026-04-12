from roboflow import Roboflow

rf = Roboflow(api_key="VkuB23r7bC6TbsLHdmpU")
project = rf.workspace("rakib-rayans").project("face-shield-detection-0znbd")
version = project.version(1)
dataset = version.download("yolov8", location="dataset")