import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_G
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np
import os
from glob import glob
import matplotlib.cm as cm

from demo.base_func import get_match, make_matching_plot_fast

warnings.filterwarnings("ignore")

def draw_matches(im_A, kpts_A, im_B, kpts_B):    
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, 
                    matches_A_to_B, None)
    return ret

def inner_matrix_resize(cameraMatrix, ori_image_shape, resize_image_shape):
    oh, ow = ori_image_shape
    rh, rw = resize_image_shape
    scale_x = rw / ow
    scale_y = rh / oh

    k00 = cameraMatrix[0, 0] * scale_x
    k11 = cameraMatrix[1, 1] * scale_y
    k02 = cameraMatrix[0, 2] * scale_x
    k12 = cameraMatrix[1, 2] * scale_y
    cameraMatrix_resize = np.array([[k00, 0.0, k02], [0.0, k11, k12], [0.0, 0.0, 1.0]])
    return cameraMatrix_resize

if __name__ == "__main__":
    device = get_best_device()
    detector = dedode_detector_L(weights = torch.load(r"G:\point_match\DeDoDe\weights\dedode_detector_L_v2.pth", map_location = device))
    descriptor = dedode_descriptor_G(weights = torch.load(r"G:\point_match\DeDoDe\weights\dedode_descriptor_G.pth", map_location = device))
    matcher = DualSoftMaxMatcher()


    data_path = r'G:\point_match\calibrate\camera_test_gt_val\test_30'
    psave_dir = os.path.join(data_path, 'spsg_process_save_dir')
    os.makedirs(psave_dir, exist_ok=True)

    K_20 = np.array( [[1.50654105e+03, 0.00000000e+00, 9.52765666e+02],
                        [0.00000000e+00, 1.50409791e+03, 5.27824348e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_20 = np.array([[3.59716101e-02, -3.86153710e-01, 8.50330946e-05, -4.09732074e-04, 1.29951151e-01]])

    K_30 = np.array([[1.35121009e+03, 0.00000000e+00, 1.95377803e+03],
                     [0.00000000e+00, 1.34432810e+03, 1.13253609e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_30 = np.array([[-0.01668452, -0.01945304, -0.00125963, -0.00154738,  0.00329328]])

    K_resize = inner_matrix_resize(K_30, (2160, 3840), (480, 640))
    fx, fy, cx, cy = K_resize[0, 0], K_resize[1, 1], K_resize[0, 2], K_resize[1, 2]

    files = ['1_2','2_3','3_4','4_1'] # ,
    angle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


    for file in files:
        data_path_file = os.path.join(data_path,file)
        for angle_i in angle:
            file_path = os.path.join(data_path_file, str(angle_i))
            for item in os.listdir(file_path):
                impath = os.path.join(file_path, item)
                print(f'{angle_i},{item}')
                externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
                image_files = list()
                for extern in externs:
                    image_files.extend(glob(impath + "\\*." + extern))
                print(f'{image_files[0]}-{image_files[1]},GT:[{item}]')
                assert len(image_files) == 2

                save_dir = os.path.join(impath, 'dedode')
                os.makedirs(save_dir, exist_ok=True)

                im_A_path = image_files[0]
                im_B_path = image_files[1]

                im_A = Image.open(im_A_path)
                im_B = Image.open(im_B_path)
                target_size = (640,480)
                im_A = im_A.resize(target_size, Image.BILINEAR)
                im_B = im_B.resize(target_size, Image.BILINEAR)

                shape_im_A = im_A.size
                shape_im_B = im_B.size
                print(f'last_frame_ori shape: {shape_im_A},last_frame_color shape: {shape_im_B} \n K: {K_resize}')

                W_A, H_A = im_A.size
                W_B, H_B = im_B.size

                detections_A = detector.detect_from_path(im_A_path, num_keypoints = 10_000)
                keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
                detections_B = detector.detect_from_path(im_B_path, num_keypoints = 10_000)
                keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]
                description_A = descriptor.describe_keypoints_from_path(im_A_path, keypoints_A)["descriptions"]
                description_B = descriptor.describe_keypoints_from_path(im_B_path, keypoints_B)["descriptions"]
                matches_A, matches_B, batch_ids = matcher.match(keypoints_A, description_A,
                    keypoints_B, description_B,
                    P_A = P_A, P_B = P_B,
                    normalize = True, inv_temp=20, threshold = 0.01)#Increasing threshold -> fewer matches, fewer outliers

                matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)
                kp1 = matches_A.detach().cpu().numpy()
                kp2 = matches_B.detach().cpu().numpy()
                np.save(fr'{save_dir}\kp1.npy', kp1)
                np.save(fr'{save_dir}\kp2.npy', kp2)
                #
                # Image.fromarray(draw_matches(im_A, matches_A, im_B, matches_B)).save(r"G:\point_match\calibrate\camera_test_gt_val\50_10\matches.jpg")
                im_A_np = np.array(im_A)
                im_B_np = np.array(im_B)

                text = [
                    'DeDoDe',
                    'Keypoints: {}:{}'.format(len(keypoints_A), len(keypoints_B)),
                    'Matches: {}'.format(len(kp1))
                ]
                small_text = [
                    # 'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    # 'Match Threshold: {:.2f}'.format(m_thresh),
                    # 'Image Pair: {:06}:{:06}'.format(stem0, stem1),
                ]

                get_match(im_A_np, im_B_np, K_resize, save_dir,
                          keypoints1=kp1, keypoints2=kp2, match_name='match_points_resize',
                          name='reproject3d_resize_de', refine=False)

                color = cm.jet([1]*len(kp1))
                out = make_matching_plot_fast(
                    im_A_np, im_B_np, keypoints_A, keypoints_B, kp1, kp2, color, text,
                    path=None, show_keypoints=False, small_text=small_text, margin=0)
                cv2.imwrite(f"{save_dir}/resized_match_points_de.jpg", out)


