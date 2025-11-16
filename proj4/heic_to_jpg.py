from PIL import Image
from pillow_heif import register_heif_opener
import os
from pathlib import Path

register_heif_opener()
files = os.listdir("./heic")
os.makedirs(Path("nagi_images2"), exist_ok=True)
for file in files:
    file_name_with_extension = os.path.basename(file)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    image = Image.open(os.path.join("heic", file)).convert("RGB")
    image = image.resize((200, int(4 / 3 * 200)))
    image.save(
        Path("nagi_images2") / (file_name_without_extension + ".jpg"),
        "JPEG",
        quality=95,
    )


image = Image.open("heic/IMG_2646.HEIC")
image = image.resize((1000, int(4 / 3 * 1000)))
image.save("nagi_image.jpg", "JPEG", quality=95)
