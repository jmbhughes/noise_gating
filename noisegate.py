import numpy as np

fft = np.fft.fftn
ifft = np.fft.ifftn


def define_image_coordinates(data, width=12):
    x_width, y_width, t_width = width, width, width

    # define grid
    x_start, x_end, x_step = x_width//2, data.shape[0] - (x_width//2), 3
    y_start, y_end, y_step = y_width//2, data.shape[1] - (y_width//2), 3
    t_start, t_end, t_step = t_width//2, data.shape[2] - (t_width//2), 3

    x_ = np.arange(x_start, x_end, x_step)
    y_ = np.arange(y_start, y_end, y_step)
    t_ = np.arange(t_start, t_end, t_step)
    coordindates = []
    for x in x_:
        for y in y_:
            for t in t_:
                coordindates.append((x, y, t))
    return coordindates


def define_hanning_window(width=12):
    x_width, y_width, t_width = width, width, width

    # equation for a hanning window
    hanning_window_function = lambda x, y, t: (np.power(np.sin((x + 0.5) * np.pi / x_width), 2.0) *
                                         np.power(np.sin((y + 0.5) * np.pi / y_width), 2.0) *
                                         np.power(np.sin((t + 0.5) * np.pi / t_width), 2.0))

    # set up our hanning_window  array
    hanning_window = np.zeros((x_width, y_width, t_width))
    for x in range(hanning_window.shape[0]):
        for y in range(hanning_window.shape[1]):
            for t in range(hanning_window.shape[2]):
                hanning_window[x, y, t] = hanning_window_function(x, y, t)
    return hanning_window


def calculate_beta(data, coordinates, count=10000, percentile=50, width=12):
    xwidth, ywidth, twidth = width, width, width

    beta_stack = []

    # for count random regions of the image cube we will take the fourier transform
    # and use it to estimate the noise model, beta
    for i in np.random.choice(len(coordinates), count):
        x, y, t = coordinates[i]

        # get the data
        image_section = data[x - xwidth // 2: x + xwidth // 2,
                        y - ywidth // 2: y + ywidth // 2,
                        t - twidth // 2: t + twidth // 2].copy()
        # image_section *= hanning # multiply by hanning window

        # sum of the square root of the image section
        imbar = np.sum(np.sqrt(image_section))

        # take the fourier transform and get the magnitude
        beta_fourier = fft(image_section)
        beta_fourier_magnitude = np.abs(beta_fourier)

        # add this beta term to the stack
        beta_stack.append(beta_fourier_magnitude / imbar)

    # from all the sections use the specified percentile
    beta_approx = np.nanpercentile(np.stack(beta_stack), percentile, axis=0)
    return beta_approx


def noise_gate(data, coords, beta_approx, gamma=3, width=12):
    apod_scale = 1 / 2.3  # I have lost track of where this number came from, but it appears to work on SUVI data

    xwidth, ywidth, twidth = width, width, width
    hanning = define_hanning_window(width=width)
    gated_image = np.zeros_like(data)

    # over all image sections
    for i in range(len(coords)):
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
        fourier = fft(image_section)
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
        final_image = apod_scale * hanning * np.abs(ifft(final_fourier))

        # add into the final gated image
        gated_image[x - xwidth // 2: x + xwidth // 2,
        y - ywidth // 2: y + ywidth // 2,
        t - twidth // 2: t + twidth // 2] += final_image

    return gated_image
