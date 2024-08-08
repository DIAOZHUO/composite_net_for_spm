from nets.topography_composite_net.model import CompositeNet

model = CompositeNet()
model.save(save_path="./composite_net.pt",
           overwrite_net1_path="../tip_quality_classification/output/model20231121-181401_train_param.pth",
           overwrite_net2_path="../si77_adsorption_detection/runs/detect/train/weights/best.pt",
           overwrite_net3_path="../si77_kp_detection/runs/pose/train2/weights/best.pt")


