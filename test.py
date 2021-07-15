import io
from PIL import Image
from src.utils.utils import save_image_to_s3

im = Image.open(r"C:\Users\Nhat Thanh\Desktop\public\images\64d0f3b3-ffc0-4f8c-999e-9a968fc51292.jpg")
im_byte_arr = io.BytesIO()
im.save(im_byte_arr, format='PNG')
im_byte_arr = im_byte_arr.getvalue()
url = save_image_to_s3(im_byte_arr, 'my_image.jpg')
print(url)