import argparse
import numpy as np
import skimage.util


def get_noise_model(noise_type="gaussian,0.15"):
    tokens = noise_type.split(sep=",")

    if tokens[0] == "gaussian":

        def gaussian_noise(img):
            noise_img = skimage.util.random_noise(img, var=float(tokens[1])**2)
            noise_img *= 255
            return noise_img.astype('uint8')
        return gaussian_noise
    elif tokens[0] == "clean":
        return lambda img: img
    elif tokens[0] == "impulse":
        min_occupancy = int(tokens[1])
        max_occupancy = int(tokens[2])

        def add_impulse_noise(img):
            occupancy = np.random.uniform(min_occupancy, max_occupancy)
            mask = np.random.binomial(size=img.shape, n=1, p=occupancy / 100)
            noise = np.random.randint(256, size=img.shape)
            img = img * (1 - mask) + noise * mask
            return img.astype(np.uint8)
        return add_impulse_noise
    else:
        raise ValueError("noise_type should be 'gaussian', 'clean', or 'impulse'")


def get_args():
    parser = argparse.ArgumentParser(description="test noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--noise_model", type=str, default="gaussian,0,50",
                        help="noise model to be tested")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_size = args.image_size
    noise_model = get_noise_model(args.noise_model)

    while True:
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128
        noisy_image = noise_model(image)

        # "q": quit
        if key == 113:
            return 0


if __name__ == '__main__':
    main()
