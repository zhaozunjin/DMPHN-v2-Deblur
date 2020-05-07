## Title
Improved Deep Multi-Patch Hierarchical Network with Nested Module for Dynamic Scene Deblurring

## Abstract
Dynamic scene deblurring is a significant technique in the field of computer vision. The multi-scale strategy has been successfully extended to the deep end-to-end learning-based deblurring task. Its expensive computation gives birth to the multi-patch framework. The success of the multi-patch framework benefits from the local residual information passed across the hierarchy. One problem is that the finest levels rarely contribute to their residuals so that the contributions of the finest levels to their residuals are excluded by coarser levels, which limits the deblurring performance. To this end, we substitute the nested module blocks, whose powerful and complex representation ability is utilized to improve the deblurring performance, for the building blocks of the encoder-decoders in the multi-patch network. Additionally, the attention mechanism is introduced to enable the network to differentiate blur across the whole blurry image from dynamic scene, thereby further improving the ability to handle the motion object blur. Our modification boosts the contributions of the finest levels to their residuals and enables the network to learn different weights for feature information extracted from spatially-varying blur image. Extensive experiments show that the improved network achieves competitive performance on the GoPro dataset according to PSNR and SSIM.

## Performance


## Install
- torch-1.3.0
- torchvision
- numpy
- scipy
- Pillow
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

[Download the model parameter files.](https://drive.google.com/drive/folders/1aVeJ_GbBTM-Q0oaxpIH5TWS8wW48Qadf)

- **train**
```
python xxx.py -b 6
```
As an improved version of DMPHN, to deomstrate the better performance of the proposed model, we adopted the (1-2-4-8) multi-patch manner in our experiments since DMPHN got the best performance by the manner. You can import different model in the folder named models like this：
```
import models.modelstitus as models
```
- **test**
```
python xxx_test.py
```
Before you run this program, please insure that the model parameters are downloaded and saved in the folder named checkpoints, and then you can find the resulting images and the intermediate residual images.
