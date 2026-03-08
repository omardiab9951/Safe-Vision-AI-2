from roboflow import Roboflow

rf = Roboflow(api_key="VkuB23r7bC6TbsLHdmpU")

project = rf.workspace("arman-keresh-lbrre").project("vest-no-vest")

version = project.version(1)

dataset = version.download("yolov8")