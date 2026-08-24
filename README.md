# RDBD_INTERN_ASSESSMENT_HASINA
In this section, the related works in traffic sign detection are discussed and the major 
techniques, utilized datasets, problems are summarized. We focus on both classical CNN based 
models and modern light/hybrid architectures which are more appropriate for real-time 
realization as well as for Bangladesh context. 
Mareeswari et al. (2025) [6] also compare YOLOv8 on the dataset TT100K, consisting of over 
100k annotated traffic sign images. The authors also compared YOLOv8 with both YOLOv7 
and YOLOv5 demonstrating an accuracy of about 94%, which was much higher than that of 
former versions. Although the model achieved good accuracy, the accuracy saturated at 94%, 
that might be a limitation for safety-critical scenarios such as autonomous driving in complex 
environments. 
Luo et al. (2024) [7] introduced an enhanced model, YOLOv8-CE (YOLOv8 with Coordinate 
Attention and Enhanced IoU). For TT100K, the model obtained 96% accuracy with an 
inference time of 101 ms, demonstrating real-time capabilities. Nonetheless, the increased 
architectural sophistication comes at a cost of higher computational requirements, which may 
limit applications for resource-constrained embedded devices. 
Renuka (2024) [8] fine-tuned YOLOv8 on Kaggle Bangladeshi traffic sign dataset, which is a 
multiclass dataset. The model also achieved 80.6% accuracy with 65.7% recall. The authors 
reported difficulties in the detection of small, blurred or occluded signs typical on Bangladeshi 
road scenes. This indicates that balancing the dataset and augmentations are crucial. 
Farid & Islam (2019) [9] unveiled a Bangladeshi Traffic Sign Dataset containing 10,259 real
world images in 31 classes. For their baseline CNN-and YOLO-based models, their accuracy 
was under 97%, while the dataset had class imbalance and environmental noise such that model 
generalization is poor in diverse conditions. 
Chen & Fan (2024) [10] proposed MSGC-YOLOv8 is optimized for detecting traffic signs in 
snow. Their model further enhanced mAP by approximately 18% with a staggering of ~60% 
reduction of parameters compared to YOLOv8n/s, by incorporating the multi-scale group 
4 
convolutions, deformable attention and some extra detection layers; nevertheless the study was 
only presented in snow scenes making parasagophile’s generalization over fog/ lowlight 
conditions unknown. 
Qu et al. (2023) [11] proposed to modify YOLOv5 by adding coordinate attention and 
introducing an extra prediction head for detecting small objects. Trained on the CCTSDB 2021 
dataset (≈25k signs), it achieved 88.1% precision and 79.8% recall in adverse-weather 
conditions. However, % on cars), resulting in suboptimal robustness to clutter. 
Youness & Soumia (2025) [12] proposed RoadNet : a light-weighted CNN for Moroccan traffic 
signs making multi-scale feature extraction and transfer learning. With 96% as the training 
accuracy and 88.6% the validation accuracy, RoadNet surpasses VGG16 with efficiency for 
real-time operation. 
Chu et al. [13] proposed a method which utilized global features by self-attention and a 
lightweight decoupled parallel detection head. Reasonable results were also obtained on 
TT100K: the model yielded 88.8% accuracy and with a recall rate of 83.2%, exhibiting some 
progress in small object recognition. 
Alawaji et al. (2024) [14] studied the multi-task learning for traffic sign and signal recognition 
with InceptionResNetV2 and DenseNet201 backbones along with ROI modules. Evaluated on 
highways in Riyadh, the system obtained 99.07% accuracy using real time process. 
Reddy et al. (2025) [15] proposed a comparison of several deep learning models for adaptive 
traffic sign detection such as ResNet, Inception, MobileNet, Darknet or R-FCN ResNet101, 
SSD MobileNet and YOLOv2 trained on MS COCO. It was found that R-FCN ResNet101 
achieved the best trade-off between precision and speed, SSD MobileNet was also efficient in 
terms of memory and real-time capacity and YOLOv2 obtained the highest precision accuracy, 
making it specific for embedded platforms. 
Hong et al. (2025) [16] proposed YOLO-BS, an enhanced YOLOv8-based model for small 
signs detection in complex background. With the inclusion of a small-object detection layer 
and BiFPN, YOLO-BS achieved 90.1% mAP50 and 78 FPS on TT100K as well as better 
performance compared to other state-of-the-art models. 
