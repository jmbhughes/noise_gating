import argparse
import glob
from astropy.io import fits
import numpy as np
from noisegate import define_image_coordinates, calculate_beta, noise_gate
import os


def load_data(path):
    fns = sorted(glob.glob(path + "*.fits"))
    data = []
    for i, fn in enumerate(fns):
        with fits.open(fn) as image_file:
            img = image_file[0].data

            # make sure all the entries are valid
            img[np.isnan(img)] = 0
            img[np.isinf(img)] = 0
            img[img < 0] = 0
            data.append(img)
    data = np.stack(data, axis=2)
    return fns, data


def save_data(gated_cube, fns, out_directory):
    for i, fn in enumerate(fns):
        new_fn = os.path.join(out_directory, os.path.basename(fn.replace(".fits", "_gated.fits")))
        with fits.open(fn) as old_fits:
            header = old_fits[0].header
            fits.writeto(new_fn, gated_cube[:, :, i], header)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Noise gate an image sequence")
    parser.add_argument("in_directory", type=str, help="location of FITS images to reprocess")
    parser.add_argument("out_directory", type=str, help="where to save the noise gated FITS images")
    parser.add_argument("--gamma", type=float, default=3.0,
                        help="ad hoc bias factor to reject noise, see eqn 12 in DeForest (2017)")
    parser.add_argument("--beta_percentile", type=float, default=50.0,
                        help="what percentile to use when calculating beta")
    parser.add_argument("--beta_count", type=int, default=10000,
                        help="how many image tiles to use when calculating beta")
    parser.add_argument("--width", type=int, default=12, help="size of the window to use for FFT tiles")
    args = parser.parse_args()

    fns, data = load_data(args.in_directory)
    coordinates = define_image_coordinates(data, width=args.width)
    beta = calculate_beta(data, coordinates, width=args.width, count=args.beta_count, percentile=args.beta_percentile)
    gated = noise_gate(data, coordinates, beta, gamma=args.gamma, width=args.width)
    save_data(gated, fns, args.out_directory)