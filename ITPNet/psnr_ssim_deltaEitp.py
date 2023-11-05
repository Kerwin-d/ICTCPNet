import os
import cv2
import numpy as np
from tqdm import tqdm
import colour


def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def _ssim(img, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_hdr_deltaITP(img1, img2):
    img1 = colour.models.eotf_ST2084(img1)
    img2 = colour.models.eotf_ST2084(img2)
    img1_ictcp = colour.RGB_to_ICTCP(img1)
    img2_ictcp = colour.RGB_to_ICTCP(img2)
    delta_ITP = 720 * np.sqrt((img1_ictcp[:,:,0] - img2_ictcp[:,:,0]) ** 2
                            + 0.25 * ((img1_ictcp[:,:,1] - img2_ictcp[:,:,1]) ** 2)
                            + (img1_ictcp[:,:,2] - img2_ictcp[:,:,2]) ** 2)
    return np.mean(delta_ITP)


def compute_psnr(gt, hdr, peak=1.0):
    mse = np.mean(np.square(gt - hdr))
    psnr = 10 * np.log10(peak * peak / mse)
    return psnr


def PSNR(HDR_path, metrics):
    GT_path = r'test_set/test_hdr'

    name_list = os.listdir(HDR_path)
    psnr_all = []
    ssim_all = []
    deltaE_ITP_sum = []
    for name in tqdm(sorted(name_list)):
        GT_file = GT_path + '/' + name
        HDR_file = HDR_path + '/' + name
        GT = np.array(cv2.imread(GT_file, flags=-1), np.float32)[:, :, ::-1] / 65535
        HDR = np.array(cv2.imread(HDR_file, flags=-1), np.float32)[:, :, ::-1] / 65535

        #psnr
        if "psnr" in metrics:
            psnr = compute_psnr(GT, HDR)
            psnr_all.append(psnr)

        #ssim
        if "ssim" in metrics:
            ssim = calculate_ssim(GT*255, HDR*255)
            ssim_all.append(ssim)

        #deltaE_ITP
        if "deltaE_ITP" in metrics:
            deltaE_ITP = calculate_hdr_deltaITP(GT, HDR)
            deltaE_ITP_sum.append(deltaE_ITP)

    print(HDR_path)
    print("PSNR: ", np.mean(psnr_all))
    print("SSIM: ", np.mean(ssim_all))
    print("deltaE_ITP: ", np.mean(deltaE_ITP_sum))



if __name__ == '__main__':
    path = r'result'
    metrics = ["psnr", "ssim", "deltaE_ITP"]
    PSNR(path, metrics)
