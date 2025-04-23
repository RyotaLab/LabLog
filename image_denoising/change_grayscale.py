from PIL import Image

# PNG画像を開く
image = Image.open("image_denoising/nnoised_circle.png")

# グレースケールに変換
grayscale_image = image.convert("L")

# 変換した画像を保存
grayscale_image.save("image_denoising/noised_circle.png")

# 表示（必要に応じて）
grayscale_image.show()