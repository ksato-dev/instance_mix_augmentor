import cv2
import numpy as np
import random

def overlay_image(bg_img_path, overlay_img_path, x=None, y=None):
    # 背景画像を読み込む
    bg_img = cv2.imread(bg_img_path)

    # 透過画像を読み込む
    overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)

    # 背景画像のサイズを取得
    bg_height, bg_width = bg_img.shape[:2]

    # 透過画像のサイズを取得
    overlay_height, overlay_width = overlay_img.shape[:2]

    # 指定が無ければ x, y の範囲ランダムで決定
    if x is None:
        x = random.randint(0 + overlay_width, bg_width - overlay_width)
    if y is None:
        y = random.randint(0 + overlay_height, bg_height - overlay_height)

    # 透過画像をRGBAに変換
    overlay_img = overlay_img

    # 透過画像のアルファチャンネルを抽出
    alpha = overlay_img[:, :, 3] / 255.0

    # 背景画像の該当部分を切り出す
    bg_roi = bg_img[y:y+overlay_height, x:x+overlay_width]

    # 透過画像を背景画像に重ねる
    for c in range(3):
        bg_roi[:, :, c] = (1 - alpha) * bg_roi[:, :, c] + alpha * overlay_img[:, :, c]

    # 結果を保存
    cv2.imwrite('output.jpg', bg_img)

# 座標（x, y）を設定
x = 100
y = 50

# 画像のパスを設定
bg_img_path = 'background.jpg'
overlay_img_path = 'overlay/obj001.png'

# 画像を重ねる関数を実行
# overlay_image(bg_img_path, overlay_img_path, x, y)
overlay_image(bg_img_path, overlay_img_path)
