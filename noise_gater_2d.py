

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

fft = np.fft.fft2
ifft = np.fft.ifft2
xwidth, ywidth, twidth = 20, 20, 0

if __name__ == "__main__":    
    filename = "/Users/mhughes/google_drive/codedungeon/sun_selector/data/thm_files/set_10/OR_SUVI-L1b-Fe195_G16_s20170911037044_e20170911037054_c20170911037188.fits"
    outfile = "/Users/mhughes/Desktop/gated.fits"
    image_file = fits.open(filename)
    data = image_file[0].data
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0

    print("grid")
    # define grid
    xstart, xend, xstep = xwidth*2, data.shape[0] - (xwidth*2), 5
    ystart, yend, ystep = ywidth*2, data.shape[1] - (ywidth*2), 5
    x_ = np.arange(xstart, xend, xstep)
    y_ = np.arange(ystart, yend, ystep)

    coords = []
    for x in x_:
        for y in y_:
            coords.append((x,y))

    image_sections = []
    imbar_sections = []

    NX, NY = 2*xwidth+1, 2*ywidth+1
    hanning_window_2D = lambda x, y : np.power(np.sin((x + 0.5)*np.pi / NX), 2.0) * np.power(np.sin((y + 0.5) * np.pi / NY), 2.0)

    hanning = np.zeros((NX, NY))
    for x in range(hanning.shape[0]):
        for y in range(hanning.shape[1]):
            hanning[x,y] = hanning_window_2D(x,y)

    print("image section retrieved")
    for coord in coords:
        x, y = coord
        image_section = data[x-xwidth : x+xwidth + 1, 
                              y-ywidth : y+ywidth + 1].copy()
        #image_section = signal.convolve(image_section, hanning, mode='same')
        image_section *= hanning
        image_sections.append(image_section)#, t-twidth//2])
        imbar_sections.append(np.sum(np.sqrt(image_section)))

    image_fourier_sections = []
    image_fourier_sections_magnitude = []
    for section in image_sections:
        image_fourier = fft(section)
        image_fourier_sections.append(image_fourier)
        image_fourier_sections_magnitude.append(np.log(np.abs(np.fft.fftshift(image_fourier))**2))

    beta_approx_fn = lambda kx, ky: np.median([image_fourier_sections_magnitude[i][kx,ky] / imbar_sections[i] for i in range(len(image_sections))])
    beta_approx = np.zeros_like(image_sections[0])
    for kx in range(beta_approx.shape[0]):
        for ky in range(beta_approx.shape[1]):
            beta_approx[kx,ky] = beta_approx_fn(kx, ky)

    noise_sections = [beta_approx * imbar_sections[i] for i in range(len(image_sections))]
    threshold = [3 * noise_section for noise_section in noise_sections]

    gate_filter = [image_fourier_sections_magnitude[i] > threshold[i] for i in range(len(image_sections))]
    #wiener_filter =  [(image_fourier_sections_magnitude[i]  / threshold[i]) / (1 + (image_fourier_sections_magnitude[i]  / threshold[i])) for i in range(len(image_sections))]

    final_fourier_sections = [image_fourier_sections[i] * gate_filter[i] for i in range(len(image_sections))]

    new_image = np.zeros_like(data)
    for i in range(len(coords)):
        x, y = coords[i]
        new_image[x-xwidth : x+xwidth + 1, 
                  y-ywidth : y+ywidth + 1] += hanning * np.abs(ifft(final_fourier_sections[i]))

    f, axarr = plt.subplots(2)
    axarr[0].imshow(new_image)
    axarr[1].imshow(data)
    plt.show()

    image_file[0].data = new_image
    image_file.writeto(outfile)
