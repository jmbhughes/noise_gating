from astropy.io import fits
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import argparse

def wiener_filter(magnitude, threshold):
    term = magnitude / threshold
    return term / (1 + term)

class NoiseGater:
    def __init__(self, image_cube, beta, xwidth=12, ywidth=12, twidth=12,
                 xstep=3, ystep=3, tstep=3,
                 gamma=3, beta_percentile=None, beta_count=None):
        self.image_cube = image_cube
        self.xwidth = xwidth
        self.ywidth = ywidth
        self.twidth = twidth
        self.xstep = xstep
        self.ystep = ystep
        self.tstep = tstep
        self.gamma = gamma
        self.coordinates = self._build_coordinates()
        self.hanning_window = self._build_hanning_window()
        if beta == None:
            self.beta = self.calculate_beta(beta_percentile, beta_count)
        else:
            self.beta = beta

    def calculate_beta(self, beta_percentile=50, beta_count=1000):
        beta_stack = []
        for i in np.random.choice(len(self.coordinates), beta_count):
            x, y, t = self.coordinates[i]
            image_section = self.image_cube[x-self.xwidth//2 : x+self.xwidth//2,
                                            y-self.ywidth//2 : y+self.ywidth//2,
                                            t-self.twidth//2 : t+self.twidth//2].copy()
            image_section *= self.hanning_window
            imbar = np.sum(np.sqrt(image_section))
            fourier = np.fft.fftn(image_section)
            fourier_magnitude = np.abs(fourier)
            beta_stack.append(fourier_magnitude / imbar)
        result = np.percentile(np.stack(beta_stack), beta_percentile, axis=0)
        del beta_stack
        return result
    
    def _build_coordinates(self):
        ''' Determine center coordinates of all image sections in the image cube ''' 
        xstart, xend, xstep = self.xwidth*2, self.image_cube.shape[0] - (self.xwidth*2), self.xstep
        ystart, yend, ystep = self.ywidth*2, self.image_cube.shape[1] - (self.ywidth*2), self.ystep
        tstart, tend, tstep = self.twidth*2, self.image_cube.shape[2] - (self.twidth*2), self.tstep
        x_ = np.arange(xstart, xend, xstep)
        y_ = np.arange(ystart, yend, ystep)
        t_ = np.arange(tstart, tend, tstep)
        coordinates = []
        for x in x_:
            for y in y_:
                for t in t_:
                    coordinates.append((x,y,t))
        return coordinates

    def _build_hanning_window(self):
        ''' Calculate  a hanning window with the appropriate size '''
        hanning_function = lambda x, y, t : (np.power(np.sin((x + 0.5) * np.pi / self.xwidth), 2.0) * 
                                             np.power(np.sin((y + 0.5) * np.pi / self.ywidth), 2.0) * 
                                             np.power(np.sin((t + 0.5) * np.pi / self.twidth), 2.0))
        hanning = np.zeros((self.xwidth, self.ywidth, self.twidth))
        for x in range(hanning.shape[0]):
            for y in range(hanning.shape[1]):
                for t in range(hanning.shape[2]):
                    hanning[x,y,t] = hanning_function(x,y,t)
        return hanning
    
    def _process_section(self, i):
        x, y, t, = self.coordinates[i]
        image_section = self.image_cube[x-self.xwidth//2 : x+self.xwidth//2,
                                        y-self.ywidth//2 : y+self.ywidth//2,
                                        t-self.twidth//2 : t+self.twidth//2].copy()
        image_section *= self.hanning_window
        imbar = np.sum(np.sqrt(image_section))
        fourier = np.fft.fftn(image_section)
        fourier_magnitude = np.abs(fourier)
        noise = self.beta * imbar
        threshold = noise * self.gamma
        #section_filter = se[self.filter_fn](fourier_magnitude, threshold)
        section_filter = wiener_filter(fourier_magnitude, threshold)
        final_fourier = fourier * section_filter
        final_image = self.hanning_window * np.abs(np.fft.ifftn(final_fourier))
        del image_section
        del fourier_magnitude
        del final_fourier
        return final_image

    def clean(self):
        #self.coordinates = self._build_coordinates()
        #self.hanning_window = self._build_hanning_window()
        final_image = np.zeros_like(self.image_cube)
        for i in range(len(self.coordinates)):
            section = self._process_section(i)
            x, y, t = self.coordinates[i]
            final_image[x-self.xwidth//2 : x+self.xwidth//2,
                        y-self.ywidth//2 : y+self.ywidth//2,
                        t-self.twidth//2 : t+self.twidth//2] += section
        return final_image

    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required = True, help = "file with path to FITs images for sequence")
    ap.add_argument("-b", "--beta", help = "image cube of appropriate size with beta image")
    ap.add_argument("-g", "--gamma", help = "gamma level to define threshole for noise")
    ap.add_argument("-v", "--verbose", help = "prints informatino for each step")
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    args = get_args()

    if args['verbose']:
        print("Opening files")
        
    with open(args['files']) as f:
        files = f.readlines()
    image_stack = []
    for fn in files:
        image = fits.open(fn.split("\n")[0])
        image_stack.append(image[0].data.copy())
        image.close()

    if args['verbose']:
        print("Making image cube")
    image_cube = np.stack(image_stack)

    if args['verbose']:
        print("Cleaning image")
    ng = NoiseGater(image_cube, None, beta_percentile=50, beta_count=10000)
    clean_cube = ng.clean() 
    
