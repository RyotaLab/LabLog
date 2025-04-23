from PIL import Image
import numpy as np

image_file = "/Users/tanabou/授業系/seminar/seminar_to_github/test.png"
im = np.array(Image.open(image_file))


# pil_img = Image.fromarray(im)
# pil_img.save("changed_text.png")

print(im.shape)