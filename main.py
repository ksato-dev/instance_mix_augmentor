import cv2
import numpy as np
import random
import os


def overlay_image(bg_img_path, overlay_img_path_list, num_iterations=10):
    
    for iter_id in range(num_iterations):
        bg_roi = None

        # 背景画像を読み込む
        bg_img = cv2.imread(bg_img_path)
        
        num_ol_img_path_list = len(overlay_img_path_list)
        skip_flag_list = [random.randint(0, 1) for _ in range(num_ol_img_path_list)]  # 画像 ID ごとにスキップするかどうかのフラグを作る

        if sum(skip_flag_list) == num_ol_img_path_list:
            print(iter_id, skip_flag_list)
            skip_flag_list[random.randint(0, num_ol_img_path_list-1)] = 0 # 一個もオーバーレイがないならどこか一つ使わせる

        for img_id, ol_img_path in enumerate(overlay_img_path_list):
            skip_flag = skip_flag_list[img_id]
            if skip_flag == 1:
                continue

            # 透過画像を読み込む
            overlay_img = cv2.imread(ol_img_path, cv2.IMREAD_UNCHANGED)

            # 背景画像のサイズを取得
            bg_height, bg_width = bg_img.shape[:2]

            # 透過画像のサイズを取得
            overlay_height, overlay_width = overlay_img.shape[:2]

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
overlay_image(bg_img_path, overlay_img_path)
