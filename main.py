import cv2
import numpy as np
import random
import math


def overlay_image(bg_img_path, overlay_img_path_list, num_iterations=10, range_pixel_area=None):
    """
    * TODO
        - Move a lump of statement to a function.
        - Convert this function to a class.
        - Write an overview of this function.
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
            x = random.randint(0 + overlay_width, bg_width - overlay_width)
            y = random.randint(0 + overlay_height, bg_height - overlay_height)

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
bg_img_path = 'background.jpg'
overlay_img_path = [
    'overlay/obj001.png',
    'overlay/obj002.png',
    'overlay/obj003.png',
    'overlay/obj004.png', ]

# 画像を重ねる関数を実行
# overlay_image(bg_img_path, overlay_img_path, x, y)
num_iters = 10
range_area = [5000, 7000]  # [min, max]
overlay_image(bg_img_path, overlay_img_path, num_iters, range_area)
