## Title
Improved Deep Multi-Patch Hierarchical Network with Nested Module for Dynamic Scene Deblurring

[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9050555)

[webpage](https://sites.google.com/view/tituszhao/%E9%A6%96%E9%A1%B5/dmphn-v2-deblur)

## Abstract
Dynamic scene deblurring is a significant technique in the field of computer vision. The multi-scale strategy has been successfully extended to the deep end-to-end learning-based deblurring task. Its expensive computation gives birth to the multi-patch framework. The success of the multi-patch framework benefits from the local residual information passed across the hierarchy. One problem is that the finest levels rarely contribute to their residuals so that the contributions of the finest levels to their residuals are excluded by coarser levels, which limits the deblurring performance. To this end, we substitute the nested module blocks, whose powerful and complex representation ability is utilized to improve the deblurring performance, for the building blocks of the encoder-decoders in the multi-patch network. Additionally, the attention mechanism is introduced to enable the network to differentiate blur across the whole blurry image from dynamic scene, thereby further improving the ability to handle the motion object blur. Our modification boosts the contributions of the finest levels to their residuals and enables the network to learn different weights for feature information extracted from spatially-varying blur image. Extensive experiments show that the improved network achieves competitive performance on the GoPro dataset according to PSNR and SSIM.

## Install
- torch-1.3.0
- torchvision
- numpy
- scipy
- Pillow
- RAdam (https://github.com/LiyuanLucasLiu/RAdam)
- CBAM (https://github.com/luuuyi/CBAM.PyTorch)
## Dataset
- GoPro

## Experiments
- **models**

| model |  description|
|--|--|
|modelstitused| Both the encoders and decoders are consturcted by the second order nested modules.|
|modelstitusted| The encoders are consturcted by the third order nested modules, and the decoders by the second order nested modules|
|modelattentioned|The attention mechanism is introduced into the modelstitused |
|modelattentionted|The attention mechanism is introduced into the modelstituste|

[Download the model parameter files.](https://drive.google.com/open?id=1aVeJ_GbBTM-Q0oaxpIH5TWS8wW48Qadf)

- **train**
```
python xxx.py -b 6
```
As an improved version of DMPHN, to deomstrate the better performance of the proposed model, we adopted the (1-2-4-8) multi-patch manner in our experiments since DMPHN got the best performance by the manner. You can import different model in the folder named models like this：
```
import models.modelstitus as models
```
- **test**

We recommend the readers run the demo.py, since four basic models can be imported for convenience. First, you should downloaded the checkpoints and save them according the following directory structure:
checkpoints:
  -> modelstitused
      -> xxx.pkl
  -> modelstitusted
      -> xxx.pkl
  -> modelattentioned
      -> xxx.pkl
  -> modelattentionted
      -> xxx.pkl
  -> 

```
python xxx_test.py
```
Before you run this program, please insure that the model parameters are downloaded and saved in the folder named checkpoints, and then you can find the resulting images and the intermediate residual images.

## Acknowledgement
Our network architectures are based on the work of Zhang et al[1], we thank Hong-guang Zhang for answering our questions and thank again for the source code that are avilable publicly. Besides, our work gets its inspiration from [2], we take advantage of the complex representation ability of the nested modules to improve the contributions of the finest levels in the multi-patch network to their residuals. 

## Reference
[1] H. Zhang, Y. Dai, H. Li, and P. Koniusz, "Deep Stacked Hierarchical Multi-patch Network for Image Deblurring," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5978-5986. 

[2] H. Gao, X. Tao, X. Shen, and J. Jia, "Dynamic scene deblurring with parameter selective sharing and nested skip connections," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 3848-3856. 

[3] S. Woo, J. Park, J.-Y. Lee, and I. So Kweon, "Cbam: Convolutional block attention module," in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 3-19. 
