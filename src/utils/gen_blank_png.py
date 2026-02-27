from PIL import Image

# 定義圖片寬度和高度
width = 1280
height = 720

# 建立一個RGBA模式的透明圖片
# 'RGBA' 代表 紅(R)、綠(G)、藍(B) 和 透明度(A)
# 每個像素的透明度都設定為 0 (完全透明)
img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

# 儲存圖片為PNG格式
img.save('transparent_image_1280x720.png')

print(f"1280x720 的透明PNG圖片已成功建立，檔名為 transparent_image_1280x720.png")