import numpy as np
import os
import glob
import imageio


def rgb2grayscale(rgb):
    """ Convert an array from color (rgb) to grayscale. """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array(gray).astype('uint8')


def load_images(data_dir):
    """ Load images from the given paths. Paths should including filename and extension. """
    im_paths = sorted(list(glob.glob(os.path.join(data_dir, "*.jpg"))))
    ims = []

    for i, im_path in enumerate(im_paths):
        print("Loading ...{}, {:04d}".format(im_path[-49:], i), end="\r")
        im = imageio.imread(im_path)
        ims.append(im)
    print("Done. Images loaded." + 50 * " ")
    return np.array(ims).astype("float32")


def main():
    data_dir = "data/sets/line_language"

    x = [rgb2grayscale(im) for im in load_images(data_dir)]
    np.save('data/sets/line_language.npy', x)

    print('Saved dataset as .npy!')


if __name__ == '__main__':
    main()
