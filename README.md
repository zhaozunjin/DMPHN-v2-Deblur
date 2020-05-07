## Title
Improved Deep Multi-Patch Hierarchical Network with Nested Module for Dynamic Scene Deblurring

[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9050555)

[webpage](https://sites.google.com/view/tituszhao/%E9%A6%96%E9%A1%B5/dmphn-v2-deblur)

## Abstract
Dynamic scene deblurring is a significant technique in the field of computer vision. The multi-scale strategy has been successfully extended to the deep end-to-end learning-based deblurring task. Its expensive computation gives birth to the multi-patch framework. The success of the multi-patch framework benefits from the local residual information passed across the hierarchy. One problem is that the finest levels rarely contribute to their residuals so that the contributions of the finest levels to their residuals are excluded by coarser levels, which limits the deblurring performance. To this end, we substitute the nested module blocks, whose powerful and complex representation ability is utilized to improve the deblurring performance, for the building blocks of the encoder-decoders in the multi-patch network. Additionally, the attention mechanism is introduced to enable the network to differentiate blur across the whole blurry image from dynamic scene, thereby further improving the ability to handle the motion object blur. Our modification boosts the contributions of the finest levels to their residuals and enables the network to learn different weights for feature information extracted from spatially-varying blur image. Extensive experiments show that the improved network achieves competitive performance on the GoPro dataset according to PSNR and SSIM.

## Performance

\begin{table*}[]
	\setlength{\abovecaptionskip}{0pt}
	\caption{Quantitative evaluation of our models on GoPro dataset. In the table, \textit{WS} and \textit{Attention} indicate that whether parameter sharing strategy or attention mechanism is adopted.}
	\centering 
	\begin{tabular}{l|l|l|l|l|l}
		\hline
		\multicolumn{1}{c|}{Model} & \multicolumn{1}{c|}{WS} & \multicolumn{1}{c|}{Attention} & \multicolumn{1}{c|}{PSNR(dB)} & \multicolumn{1}{c|}{SSIM} & \multicolumn{1}{c}{Param(M)}\\ \hline
		\begin{tabular}[c]{@{}l@{}}Nah et al.\cite{b30}\\ Tao et al. \cite{b33}\\ DMPHN \cite{b35}\end{tabular} & \begin{tabular}[c]{@{}l@{}}-\\ -\\ -\end{tabular}  & \begin{tabular}[c]{@{}l@{}}-\\ -\\ -\end{tabular} & \begin{tabular}[c]{@{}l@{}}28.43\\ 30.25\\ 30.45\end{tabular} & \begin{tabular}[c]{@{}l@{}}0.8461\\ 0.8994\\ 0.9022\end{tabular} & \begin{tabular}[c]{@{}l@{}}303.6\\ 33.6\\ 29.0\end{tabular}       \\ \cline{1-6}
		\begin{tabular}[c]{@{}l@{}}DMPHN-v2(\#1)\\ DMPHN-v2(\#2)\end{tabular} & \begin{tabular}[c]{@{}l@{}}-\\ -\end{tabular}    & \begin{tabular}[c]{@{}l@{}}\XSolidBrush\\ \Checkmark\end{tabular}                                                   & \begin{tabular}[c]{@{}l@{}}\textbf{30.90}\\ \textbf{30.56}\end{tabular}         
		& \begin{tabular}[c]{@{}l@{}}\textbf{0.9099}\\ \textbf{0.9035}\end{tabular}          & \begin{tabular}[c]{@{}l@{}}35.2\\ 35.3\end{tabular}               \\ \cline{1-6}
		\begin{tabular}[c]{@{}l@{}}Stack-DMPHN \cite{b35}\\ Stack-DMPHN-v2(\#3)\\ Stack-DMPHN-v2(\#4)\\ Stack-DMPHN-v2(\#5)\end{tabular} & \begin{tabular}[c]{@{}l@{}}\XSolidBrush\\ \XSolidBrush\\ \Checkmark\\ \Checkmark\end{tabular} & \begin{tabular}[c]{@{}l@{}}\XSolidBrush\\ \XSolidBrush\\ \XSolidBrush\\ \Checkmark\end{tabular} &\begin{tabular}[c]{@{}l@{}}31.39\\ \textbf{31.40}\\29.84\\29.14\end{tabular}         & \begin{tabular}[c]{@{}l@{}}0.9182\\ 0.9180 \\0.8901\\0.8945 \end{tabular}& \begin{tabular}[c]{@{}l@{}}86.9\\ 86.9\\ \textbf{39.4}\\ \textbf{39.6}\end{tabular} \\ \hline
	\end{tabular}
\end{table*}

\begin{table*}[]
	\setlength{\abovecaptionskip}{0pt}
	\caption{Controlled experiments. In the table, \textit{n-s}, \textit{n-t} and \textit{r-2} indicate that second-order nested module,  third-order nested module, and 2 stacked ResBlocks are utilized, respectively. \textit{WS} and \textit{Attention} indicate that whether parameter sharing strategy or attention mechanism is adopted.}
	\centering
	\label{tab:table2}
	\begin{tabular}{llllllll}
		\hline
		\multicolumn{2}{l}{\multirow{2}{*}{Model}} & \multirow{2}{*}{Enc.} & \multirow{2}{*}{Dec.} & \multirow{2}{*}{WS} & \multirow{2}{*}{Attention} & PSNR(dB)/SSIM & \multirow{2}{*}{Param(M)} \\ \cline{7-7}
		\multicolumn{2}{l}{} &  &  &  &  & GOPR0384\_11\_00   |  Total &  \\ \hline
		DMPHN {[}35{]} &  & r-2 & r-2 & - & - & 32.23/0.9235 | 30.45/0.9022 & 29.0 \\ \hline
		\multicolumn{1}{l|}{DMPHN-v2} & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}\#1\\ \#2\\ \#3\\ \#4\end{tabular}} & \begin{tabular}[c]{@{}l@{}}n-s\\ n-t\\ n-s\\ n-t\end{tabular} & \begin{tabular}[c]{@{}l@{}}n-s\\ n-s\\ n-s\\ n-s\end{tabular} & \begin{tabular}[c]{@{}l@{}}-\\ -\\ -\\ -\end{tabular} & \begin{tabular}[c]{@{}l@{}}\XSolidBrush\\ \XSolidBrush\\ \Checkmark\\ \Checkmark\end{tabular} & \begin{tabular}[c]{@{}l@{}}\textbf{32.34/0.9246} | \textbf{30.56/0.9043}\\ \textbf{32.61/0.9275} | \textbf{30.90/0.9099}\\\textbf{32.60/0.9295} | 30.42/0.9007\\ \textbf{32.78/0.9315} | \textbf{30.56/0.9035}\end{tabular} & \begin{tabular}[c]{@{}l@{}} \textbf{29.0}\\ 35.2\\ 29.1\\ 35.3\end{tabular} \\ \hline
		\multicolumn{2}{l}{Stack-DMPHN{[}35{]}} & r-2 & r-2 & \XSolidBrush & \XSolidBrush & 33.05/0.9343 | 31.39/0.9182 & 86.9 \\ \hline
		\multicolumn{1}{l|}{Stack-DMPHN-v2} & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}\#5\\ \#6\\ \#7\\ \#8\end{tabular}} & \begin{tabular}[c]{@{}l@{}}n-s\\ n-t\\ n-s\\ n-s\end{tabular} & \begin{tabular}[c]{@{}l@{}}n-s\\ n-s\\ n-s\\ n-s\end{tabular} & \begin{tabular}[c]{@{}l@{}}\XSolidBrush\\ \XSolidBrush\\ \Checkmark\\ \Checkmark\end{tabular} & \begin{tabular}[c]{@{}l@{}}\XSolidBrush\\ \XSolidBrush\\ \XSolidBrush\\ \Checkmark\end{tabular} & \begin{tabular}[c]{@{}l@{}}32.99/0.9334 | \textbf{31.40}/0.9180\\ 32.93/0.9323 | 31.35/0.9167\\ 31.61/0.9140 | 29.84/0.8901\\ 31.57/0.9163|29.14/0.8745\end{tabular} & \begin{tabular}[c]{@{}l@{}} \textbf{86.9}\\ 105.5\\ \textbf{39.4}\\ \textbf{39.6}\end{tabular} \\ \hline
	\end{tabular}
\end{table*}

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

## Acknowledgement
Our network architectures are based on the work of Zhang et al[1], we thank Hong-guang Zhang for answering our questions and thank again for the source code that are avilable publicly. Besides, our work gets its inspiration from [2], we take advantage of the complex representation ability of the nested modules to improve the contributions of the finest levels in the multi-patch network to their residuals. 

## Reference
[1] H. Zhang, Y. Dai, H. Li, and P. Koniusz, "Deep Stacked Hierarchical Multi-patch Network for Image Deblurring," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5978-5986. 

[2] H. Gao, X. Tao, X. Shen, and J. Jia, "Dynamic scene deblurring with parameter selective sharing and nested skip connections," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 3848-3856. 

[3] S. Woo, J. Park, J.-Y. Lee, and I. So Kweon, "Cbam: Convolutional block attention module," in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 3-19. 
