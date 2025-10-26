# WSSM3
The code for the paper WSSM: A Weakly-Supervised Oral Mucosal Disease Segmentation Model Based on Multi-Task Collaboration

1. Classification Branch:
The "data" folder stores normal and abnormal oral images.
The "out_mamba" folder stores 812 CAM images (812 is the training data volume for the segmentation branch).
The "resnet50" folder stores the classification model training codes for the two models: (1) xin--fenlei_vmunet--vmamba; (2) main--net
wei_label: Combine the cam, box, and predicted results of the segmentation (the predicted results of vmunet for box training) into a pseudo label.
Generation of cam results for the two models: (1) cam--net; (2) cam_mamba--fenlei_vmunet_cam--vmamba_cam.
2. Segmentation Branch:
Training: train
Testing: tt

The document only contains the code of the paper, without the dataset. If you need it, you can contact the author via email: 2443509635@qq.com.
