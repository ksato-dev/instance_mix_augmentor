import cv2
import numpy as np
import random
import math
import sys
import glob

class InstanceMixAugmentor(object):
    def __init__(self, bg_img_path, overlay_img_path_list, num_iterations=10, range_pixel_area=None):
        self.bg_img_path_ = bg_img_path
        self.overlay_img_path_list_ = overlay_img_path_list
        self.num_iterations_ = num_iterations
        self.range_pixel_area_ = range_pixel_area

    def __skip_config(self, iter_id):
        limit = 4
        ret_sampled_ol_img_path_list = random.sample(self.overlay_img_path_list_, limit)  # limit 分だけサンプルする
        num_ol_img_path_list = len(ret_sampled_ol_img_path_list)
        ret_skip_flag_list = [random.randint(0, 1) for _ in range(num_ol_img_path_list)]  # 画像 ID ごとにスキップするかどうかのフラグを作る
    
        if sum(ret_skip_flag_list) == num_ol_img_path_list:
            print(iter_id, ret_skip_flag_list)
            ret_skip_flag_list[random.randint(0, num_ol_img_path_list-1)] = 0 # 一個もオーバーレイがないならどこか一つ使わせる

        return ret_sampled_ol_img_path_list, ret_skip_flag_list

    def __rotate_img(self, overlay_width, overlay_height, input_img):
        # 画像の中心座標を計算
        center = (overlay_width // 2, overlay_height // 2)
    
        # 回転行列を計算
        angle_deg = random.randint(0, 360)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
        # 回転後の画像サイズを計算
        radians = np.radians(angle_deg)
        sin_angle = np.sin(radians)
        cos_angle = np.cos(radians)
        new_width = int((overlay_height * np.abs(sin_angle)) + (overlay_width * np.abs(cos_angle)))
        new_height = int((overlay_height * np.abs(cos_angle)) + (overlay_width * np.abs(sin_angle)))
    
        # 新しい中心座標を計算し、回転行列を調整
        rotation_matrix[0, 2] += (new_width // 2) - center[0]
        rotation_matrix[1, 2] += (new_height // 2) - center[1]
    
        # 画像を回転させ、ゼロパディングを適用する
        ret_rotated_img = cv2.warpAffine(input_img, rotation_matrix, (new_width, new_height))
    
        ret_new_overlay_width = new_width        
        ret_new_overlay_height = new_height        
    
        return ret_new_overlay_width, ret_new_overlay_height, ret_rotated_img

    def __scale_img(self, overlay_width, overlay_height, input_img):
        alpha = input_img[:, :, 3] / 255.0
    
        # マスク部分の面積算出
        mask_img = np.zeros(alpha.shape)
        for y in range(mask_img.shape[0]):
            for x in range(mask_img.shape[1]):
                flag = alpha[y, x]
                if flag == 0:
                    continue
                mask_img[y, x] = 255
        pixel_area = cv2.countNonZero(mask_img)
    
        min_area = self.range_pixel_area_[0]
        max_area = self.range_pixel_area_[1]
    
        max_line_ratio = math.sqrt(max_area / pixel_area)
        min_line_ratio = math.sqrt(min_area / pixel_area)
    
        # 面積が min ~ max に収まるように線分比を調整
        tgt_line_ratio = random.uniform(min_line_ratio, max_line_ratio)
    
        ret_new_overlay_height = int(tgt_line_ratio * overlay_height)
        ret_new_overlay_width = int(tgt_line_ratio * overlay_width)
    
        ret_scaled_img = cv2.resize(input_img, (ret_new_overlay_width, ret_new_overlay_height))

        return ret_new_overlay_width, ret_new_overlay_height, ret_scaled_img

    def __write_coco_fmt(self):
        # TODO: Write annotated-data in coco-format.
        pass
        

    def execute(self):
        """
        * TODO
            -[x] Move a lump of statement to a function.
            -[x] Convert this function to a class.
            -[ ] Write an overview of this function.
            -[x] Add a function that rotate an image.
            -[ ] Add a function that stick instances into boundaries.
            -[ ] Add a threshold that limit overlaid images.
            -[ ] Add a function that write a label-file on coco-format.
        """
        
        for iter_id in range(self.num_iterations_):
            bg_roi = None
    
            # 背景画像を読み込む
            bg_img = cv2.imread(bg_img_path)
            
            # スキップするかどうかの設定
            sampled_ol_img_path_list, skip_flag_list = self.__skip_config(iter_id)
    
            for img_id, ol_img_path in enumerate(sampled_ol_img_path_list):
                skip_flag = skip_flag_list[img_id]
                if skip_flag == 1:
                    continue
                
                # 透過画像を読み込む
                overlay_img = cv2.imread(ol_img_path, cv2.IMREAD_UNCHANGED)
    
                # 透過画像のサイズを取得
                overlay_height, overlay_width = overlay_img.shape[:2]
    
                # 回転
                overlay_width, overlay_height, processed_overlay_img = \
                    self.__rotate_img(overlay_width, overlay_height, overlay_img)
    
                # 面積の範囲指定があればスケールする
                if self.range_pixel_area_ is not None:
                    overlay_width, overlay_height, processed_overlay_img = \
                        self.__scale_img(overlay_width, overlay_height, processed_overlay_img)
    
                # 背景画像のサイズを取得
                bg_height, bg_width = bg_img.shape[:2]
    
                # x, y の範囲ランダムで決定
                x = random.randint(0, bg_width - overlay_width)
                y = random.randint(0, bg_height - overlay_height)
    
                # 透過画像のアルファチャンネルを抽出
                alpha = processed_overlay_img[:, :, 3] / 255.0
    
                # 背景画像の該当部分を切り出す
                bg_roi = bg_img[y:y+overlay_height, x:x+overlay_width]
    
                # 透過画像を背景画像に重ねる
                for c in range(3):
                    bg_roi[:, :, c] = (1 - alpha) * bg_roi[:, :, c] + \
                        alpha * processed_overlay_img[:, :, c]
    
            # 結果を保存
            out_img_path = "output/" + str(iter_id) + ".out.jpg"
            cv2.imwrite(out_img_path, bg_img)


# 画像のパスを設定
# bg_img_path = 'background.jpg'
bg_img_path = sys.argv[1]
overlay_img_path = glob.glob(sys.argv[2] + "/*.png")

# overlay_img_path = [
#     'overlay/obj001.png',
#     'overlay/obj002.png',
#     'overlay/obj003.png',
#     'overlay/obj004.png', ]

# 画像を重ねる関数を実行
# overlay_image(bg_img_path, overlay_img_path, x, y)
num_iters = 10
# range_area = [5000, 7000]  # [min, max]
range_area = [2500, 4000]  # [min, max]
# range_area = None
instance_mix_aug = InstanceMixAugmentor(bg_img_path, overlay_img_path, num_iters, range_area)
# overlay_image(bg_img_path, overlay_img_path, num_iters, range_area)
instance_mix_aug.execute()
