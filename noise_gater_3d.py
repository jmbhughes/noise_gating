from astropy.io import fits
import numpy as np
import glob
import time
import pyfftw.interfaces
import argparse
import os

xwidth, ywidth, twidth = 12, 12, 12  # respective size of the window in the two spatial and one temporal domains


def load_images(input_path, clean=True):
    #TODO: remove defaulting size of data
    fns = sorted(glob.glob(os.path.join(input_path, "*.fits")))
    data = np.zeros((300, 300, len(fns)))  # this is set for SUVI imagery which is 1280x1280
    for i, fn in enumerate(fns):
        with fits.open(fn) as image_file:
            img = image_file[0].data

            if clean:
                # make sure all the entries are valid
                img[np.isnan(img)] = 0
                img[np.isinf(img)] = 0
                img[img < 0] = 0
                data[:, :, i] = img
    return data


def define_coordinates(data):
    # define grid
    xstart, xend, xstep = xwidth // 2, data.shape[0] - (xwidth // 2), 3
    ystart, yend, ystep = ywidth // 2, data.shape[1] - (ywidth // 2), 3
    tstart, tend, tstep = twidth // 2, data.shape[2] - (twidth // 2), 3

    x_ = np.arange(xstart, xend, xstep)
    y_ = np.arange(ystart, yend, ystep)
    t_ = np.arange(tstart, tend, tstep)
    coords = []
    for x in x_:
        for y in y_:
            for t in t_:
                coords.append((x, y, t))
    return coords


def define_hanning():
    # equation for a hanning window
    hanning_window_3D = lambda x, y, t: (np.power(np.sin((x + 0.5) * np.pi / xwidth), 2.0) *
                                         np.power(np.sin((y + 0.5) * np.pi / ywidth), 2.0) *
                                         np.power(np.sin((t + 0.5) * np.pi / twidth), 2.0))

    # set up our hanning window array
    hanning = np.zeros((xwidth, ywidth, twidth))
    for x in range(hanning.shape[0]):
        for y in range(hanning.shape[1]):
            for t in range(hanning.shape[2]):
                hanning[x, y, t] = hanning_window_3D(x, y, t)
    return hanning


def calculate_beta(data, coords, n=10000):
    beta_stack = []

    # for N random regions of the image cube we will take the fourier transform
    # and use it to estimate the noise model, beta
    for i in np.random.choice(len(coords), n):
        x, y, t = coords[i]

        # get the data
        image_section = data[x - xwidth // 2: x + xwidth // 2,
                        y - ywidth // 2: y + ywidth // 2,
                        t - twidth // 2: t + twidth // 2].copy()

        # sum of the square root of the image section
        imbar = np.sum(np.sqrt(image_section))

        # take the fourier transform and get the magnitude
        beta_fourier = pyfftw.interfaces.numpy_fft.rfftn(image_section)
        beta_fourier_magnitude = np.abs(beta_fourier)

        # add this beta term to the stack
        beta_stack.append(beta_fourier_magnitude / imbar)

    # from all the sections use the median
    beta_approx = np.nanmedian(np.stack(beta_stack), axis=0)
    return beta_approx


def noise_gate(data, coords, gamma=3):
    gated_image = np.zeros_like(data)
    times = []
    hanning = define_hanning()
    # over all image sections
    for i in range(len(coords)):
        start = time.time()
        x, y, t = coords[i]

        # get image section copy so as not to manipulate it
        image_section = data[x - xwidth // 2: x + xwidth // 2,
                        y - ywidth // 2: y + ywidth // 2,
                        t - twidth // 2: t + twidth // 2].copy()

        # adjust by hanning window
        # imbar = np.sum(np.sqrt(image_section))
        image_section *= hanning
        imbar = np.sum(np.sqrt(image_section))  # do imbar calculations go before or after the hanning filter?

        # determine fourier magnitude
        # fourier = fft(image_section)
        # myfft = pyfftw.builders.rfftn(image_section)
        # fourier = myfft()
        fourier = pyfftw.interfaces.numpy_fft.rfftn(image_section)

        fourier_magnitude = np.abs(fourier)

        # estimate noise and set the threshold to ignore
        noise = beta_approx * imbar
        threshold = noise * gamma

        # any magnitude outside this limit is noise
        gate_filter = fourier_magnitude >= threshold
        final_fourier = fourier * gate_filter

        # we can use the wiener filter instead of the gate filter here
        # wiener_filter =  (fourier_magnitude / threshold) / (1 + (fourier_magnitude / threshold))
        # final_fourier = fourier * wiener_filter

        # adjust by hanning filter again
        # myifft = pyfftw.builders.irfftn(final_fourier)

        final_image = hanning * np.abs(pyfftw.interfaces.numpy_fft.irfftn(final_fourier))

        # add into the final gated image
        gated_image[x - xwidth // 2: x + xwidth // 2,
        y - ywidth // 2: y + ywidth // 2,
        t - twidth // 2: t + twidth // 2] += final_image

        times.append(time.time() - start)

    np.save("gated_image.npy", gated_image)
    fits.writeto("gated_100.fits", gated_image[:, :, 100], overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Noisegate some images.')
    parser.add_argument("inpath", type=str, help="directory for input images")
    parser.add_argument("outpath", type=str, help="directory to output images")
    parser.add_argument("--gamma", default=3, type=float, help="noise cutoff factor", required=False)
    args = parser.parse_args()

    start = time.time()
    data = load_images(args.inpath)
    coords = define_coordinates(data)
    hanning = define_hanning()
    beta_approx = calculate_beta(data, coords)
    noise_gate(data, coords)
    end = time.time()
    print("Completed in {:.2f} seconds".format(end-start))
