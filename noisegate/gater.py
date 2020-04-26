from __future__ import annotations
import deepdish as dd
from noisegate.util import ImageLoader, HanningWindow
import numpy as np
from typing import Optional, Callable


class BetaModel:
    def __init__(self, widths: int) -> None:
        """

        :param widths:
        """
        self.widths = widths
        self.beta = None

    def fitted(self) -> bool:
        """
        :return: whether the model has been fitted
        """
        return self.beta is None

    @classmethod
    def load(cls, path: str) -> BetaModel:
        """
        Loads a BetaModel from file
        :param path: the location the model is saved, in h5 format
        :return: a new beta model
        """
        beta, widths = dd.io.load(path)
        model = BetaModel(widths)
        model.beta = beta
        return model

    def save(self, path: str) -> None:
        """
        Save the model to file for later usage
        :param path:
        :return:
        """
        dd.io.save(path, [self.beta, self.widths])

    def fit(self, img_loader, n: int = 10000, percentile: float = 50) -> None:
        """
        :param paths:
        :param img_loader:
        :param n:
        :param percentile:
        :return:
        """
        hanning_window = HanningWindow.construct([self.widths, self.widths, self.widths])

        beta_stack = []
        for patch in img_loader.sample(n=n):
            # multiply by hanning window and sum of the square root
            patch *= hanning_window
            imbar = np.sum(np.sqrt(patch))

            # take the fourier transform and get the magnitude
            beta_fourier = np.fft.fftshift(np.fft.fftn(patch))
            beta_fourier_magnitude = np.abs(beta_fourier)

            # add this beta term to the stack
            beta_stack.append(beta_fourier_magnitude / imbar)

        self.beta = np.nanpercentile(np.stack(beta_stack), percentile, axis=0)


class NoiseGater:
    def __init__(self, widths: int, beta_model: BetaModel, gamma: float = 3):
        self.widths = widths
        self.beta = beta_model.beta
        self.hanning = HanningWindow.construct([self.widths, self.widths, self.widths])
        self.gamma = gamma

    def denoise(self, loader: ImageLoader):
        fft = np.fft.fftn
        ifft = np.fft.ifftn
        gated_cube = np.zeros((loader.img_shape[0], loader.img_shape[1], loader.img_count))

        # over all image sections
        for i in range(len(loader)):
            image_section = loader.fetch(np.array([i]))[0]

            # adjust by hanning window
            image_section *= self.hanning
            imbar = np.sum(np.sqrt(image_section))

            # determine fourier magnitude
            fourier = np.fft.fftshift(fft(image_section))
            fourier_magnitude = np.abs(fourier)

            # estimate noise and set the threshold to ignore
            noise = self.beta * imbar
            threshold = noise * self.gamma

            # any magnitude outside this limit is noise
            gate_filter = np.logical_not(fourier_magnitude < threshold)
            final_fourier = fourier * gate_filter

            # adjust by hanning filter again
            final_image = self.hanning * np.abs(ifft(np.fft.ifftshift(final_fourier)))

            # add into the final gated image
            img_numbers, ii, jj = loader.loader_index_to_patch_index(np.array([i]))
            img_number, i, j = img_numbers[0], ii[0], jj[0]
            #print(i, final_image.shape)
            try:
                gated_cube[i:i+self.widths, j:j+self.widths, img_number: img_number+self.widths] += final_image
            except ValueError:
                pass
        return gated_cube
