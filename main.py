import cv2
import numpy as np
import random
import math
import sys
import glob


def overlay_image(bg_img_path, overlay_img_path_list, num_iterations=10, range_pixel_area=None):
    """
    * TODO
        -[ ] Move a lump of statement to a function.
        -[ ] Convert this function to a class.
        -[ ] Write an overview of this function.
        -[x] Add a function that rotate an image.
        -[ ] Add a function that stick instances into boundaries.
        -[ ] Add a threshold that limit overlaid images.
        -[ ] Add a function that write a label-file on coco-format.
    """
    
    for iter_id in range(num_iterations):
        bg_roi = None

        # 背景画像を読み込む
        bg_img = cv2.imread(bg_img_path)
        
        # スキップするかどうかの設定 ---
        num_ol_img_path_list = len(overlay_img_path_list)
        skip_flag_list = [random.randint(0, 1) for _ in range(num_ol_img_path_list)]  # 画像 ID ごとにスキップするかどうかのフラグを作る

        if sum(skip_flag_list) == num_ol_img_path_list:
            print(iter_id, skip_flag_list)
            skip_flag_list[random.randint(0, num_ol_img_path_list-1)] = 0 # 一個もオーバーレイがないならどこか一つ使わせる
        # ---

        for img_id, ol_img_path in enumerate(overlay_img_path_list):
            skip_flag = skip_flag_list[img_id]
            if skip_flag == 1:
                continue

            # 透過画像を読み込む
            overlay_img = cv2.imread(ol_img_path, cv2.IMREAD_UNCHANGED)

            # 透過画像のサイズを取得
            overlay_height, overlay_width = overlay_img.shape[:2]

            # 回転 ---
            # 画像の中心座標を計算
            center = (overlay_width // 2, overlay_height // 2)

            # 回転行列を計算
            angle_deg = random.randint(0, 45)
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
            overlay_img = cv2.warpAffine(overlay_img, rotation_matrix, (new_width, new_height))

            overlay_width = new_width        
            overlay_height = new_height        
            # overlay_img = cv2.warpAffine(overlay_img, rotation_matrix, (overlay_width, overlay_height))

            # # 画像を回転させる
            # overlay_img = cv2.warpAffine(overlay_img, rotation_matrix, (overlay_width, overlay_height))
            # ---

            # 面積の範囲指定があればスケールする ---
            if range_pixel_area is not None:
                alpha = overlay_img[:, :, 3] / 255.0

                # マスク部分の面積算出
                mask_img = np.zeros(alpha.shape)
                for y in range(mask_img.shape[0]):
                    for x in range(mask_img.shape[1]):
                        flag = alpha[y, x]
                        if flag == 0:
                            continue
                        mask_img[y, x] = 255
                pixel_area = cv2.countNonZero(mask_img)
                # print(img_id, pixel_area)

                min_area = range_pixel_area[0]
                max_area = range_pixel_area[1]

                max_line_ratio = math.sqrt(max_area / pixel_area)
                min_line_ratio = math.sqrt(min_area / pixel_area)

                # 面積が min ~ max に収まるように線分比を調整
                tgt_line_ratio = random.uniform(min_line_ratio, max_line_ratio)

                overlay_height = int(tgt_line_ratio * overlay_height)
                overlay_width = int(tgt_line_ratio * overlay_width)

                overlay_img = cv2.resize(overlay_img, (overlay_width, overlay_height))
            # ---

            # 背景画像のサイズを取得
            bg_height, bg_width = bg_img.shape[:2]

            # x, y の範囲ランダムで決定
            x = random.randint(0, bg_width - overlay_width)
            y = random.randint(0, bg_height - overlay_height)

            # 透過画像をRGBAに変換
            overlay_img = overlay_img

            # 透過画像のアルファチャンネルを抽出
            alpha = overlay_img[:, :, 3] / 255.0

            # 背景画像の該当部分を切り出す
            bg_roi = bg_img[y:y+overlay_height, x:x+overlay_width]

            # 透過画像を背景画像に重ねる
            for c in range(3):
                bg_roi[:, :, c] = (1 - alpha) * bg_roi[:, :, c] + \
                    alpha * overlay_img[:, :, c]

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
range_area = [2000, 2600]  # [min, max]
# range_area = None
overlay_image(bg_img_path, overlay_img_path, num_iters, range_area)
