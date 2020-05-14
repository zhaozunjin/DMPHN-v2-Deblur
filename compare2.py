from skimage import measure
import os
import cv2
path_DAI = '../DMPHN/TITUS_TEST/dai_total/' # results of DMPHN
# path_TITUS = '../DMPHN/TITUS_TEST/epoch2810_best/'
# path_TITUS = 'H:/WSDMOHNed_feat_atten_wd/epoch400/'
path_TITUS = 'H:/SDNet_ed_atten/epoch2400/'# results of the proposed method
# path1= path_TITUS+'Iter_' + str(i) + '_deblur.png'
# 测文件名为Iter_xx_deblur.png,通过设定count可测任意数量
def campare_psnr_ssim(path_TITUS):
    # 按顺序获取图像路径
    total_psnr_titus = 0
    total_psnr_dai = 0
    total_psnr_nah = 0
    total_ssim_titus = 0
    total_ssim_dai = 0
    total_ssim_nah = 0
    f = open('test_sharp_file1.txt', 'r')
    img_path_GT = f.readlines()
    f.close()
    count = len(img_path_GT)
    # count = 100 # only evaluate 100 images
    print(count)
    j=0
    sum = 0
    for i in range(0,count):
        path1 = 'Iter_' + str(i) + '_deblur.png'
        img_rgb_GT = cv2.imread('../DMPHN/'+img_path_GT[i][:-1])
        img_rgb_DAI = cv2.imread(path_DAI + path1)
        img_rgb_TITUS = cv2.imread(path_TITUS + path1)
        # print(path_GT + img_path_TITUS[i])
        # 比较psnr和ssim
        psnr_titus = measure.compare_psnr(img_rgb_GT, img_rgb_TITUS, 255)
        ssim_titus = measure.compare_ssim(img_rgb_GT, img_rgb_TITUS, data_range=255, multichannel=True)
        total_psnr_titus += psnr_titus
        total_ssim_titus += ssim_titus
        # print('Iter_' + str(i) + '_deblur.png--'+'psnr:%f'%(psnr_titus))
        # print('Iter_' + str(i) + '_deblur.png--' + 'ssim:%f' % (ssim_titus))

        psnr_dai = measure.compare_psnr(img_rgb_GT, img_rgb_DAI, 255)
        ssim_dai = measure.compare_ssim(img_rgb_GT, img_rgb_DAI, data_range=255, multichannel=True)
        total_psnr_dai += psnr_dai
        total_ssim_dai += ssim_dai

        # print('Iter_' + str(i) + '_deblur.png--' + 'psnr:%f' % (psnr_dai))
        # print('Iter_' + str(i) + '_deblur.png--' + 'ssim:%f' % (ssim_dai))

        print(path1 + ' vs ' + img_path_GT[i][:-1].split('/',1)[1])
        j += 1
        print('\tNah:%f，%f' % (psnr_titus, ssim_titus))
        # print('\ttotal_psnr_ssim:%f, %f' % (total_psnr_titus, total_ssim_titus))
        print('\tTitus:%f,%f' % (psnr_dai, ssim_dai))
        if psnr_titus>psnr_dai:
            sum += 1
    average_psnr_titus = total_psnr_titus / count
    average_psnr_dai = total_psnr_dai / count
    average_ssim_titus = total_ssim_titus / count
    average_ssim_dai = total_ssim_dai / count
    # print('\n')
    print('****************Average*********************')
    print('\tsum:%d' % sum)
    print('\tTITUS:%f，%f' % (average_psnr_titus, average_ssim_titus))
    print('\tDAI:%f,%f' % (average_psnr_dai, average_ssim_dai))


def main():
    campare_psnr_ssim(path_TITUS)

if __name__ == '__main__':
    main()