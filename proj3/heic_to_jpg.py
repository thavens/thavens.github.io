from PIL import Image
from pillow_heif import register_heif_opener
import os

register_heif_opener()
files = os.listdir("./heic")
for file in files:
    file_name_with_extension = os.path.basename(file)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    Image.open(os.path.join("heic", file)).convert("RGB").save(
        file_name_without_extension + ".jpg", "JPEG", quality=95
    )
