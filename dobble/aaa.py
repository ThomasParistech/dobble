import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_color_from_colormap(value, colormap='jet'):
    cmap = plt.get_cmap(colormap)
    rgba = cmap(value)
    # Convert RGBA to RGB
    rgb = tuple(int(rgba[i] * 255) for i in range(3))
    return rgb


def generate_images(n, size=256, font_scale=3.0, thickness=15):
    for i in range(n):
        # Create a blank white image
        img = np.ones((size, size, 3), dtype=np.uint8) * 255

        # Calculate text position
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2

        # Write text (number) on the image
        cv2.putText(img, text, (text_x, text_y), font,
                    font_scale, get_color_from_colormap(float(i)/n), thickness)

        # Save the image
        cv2.imwrite(f"number_{i}.png", img)


if __name__ == "__main__":
    n = 57
    generate_images(n)
    print(f"{n+1} images generated successfully.")
