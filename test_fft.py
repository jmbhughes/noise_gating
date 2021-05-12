from noisegate import FFTObject
import numpy as np
import time
import matplotlib.pyplot as plt

# def test_fft():
# 	n = 100
# 	x = np.linspace(0, 100, n, dtype=np.complex128)
# 	y = np.sin(x/10.0)
# 	reference = np.fft.fft(y)
#
# 	a = FFTObject(n)
# 	a.data_in[:] = y
# 	a.fft()
# 	print('reference')
# 	print(reference[:15])
# 	print('FFTW')
# 	print(a.data_out[:15])
# 	np.testing.assert_allclose(reference,a.data_out)


if __name__ == "__main__":
    n = 10
    x = np.arange(n**3, dtype=float)
    y = x.copy().reshape((n, n, n), order='C')

    a = FFTObject(n)
    d = y.copy().flatten('A')
    print(d)
    a.data_in[:] = d
    print(a.data_in[:])
    a.fft()
    print(a.data_in[:])
    #a.ifft()

    reference = np.fft.fftn(y)
    #reference = np.fft.ifftn(reference)

    layer = 0

    fig, ax = plt.subplots()
    ax.imshow(y[:, :, layer])
    ax.set_title("data")
    fig.show()

    fig, ax = plt.subplots()
    ax.imshow(np.abs(reference)[:, :, layer])
    ax.set_title("reference")
    fig.show()

    fig, ax = plt.subplots()
    ax.imshow(np.abs(a.data_out[:].reshape((n, n, n), order='C'))[:, :, layer])
    ax.set_title("actual")
    fig.show()
    # test_fft()
    #
    # n = 10000
    # x = np.linspace(0, 100, n, dtype=np.complex128)
    # y = np.sin(x/10.0)
    # reference = np.fft.fft(y)
    #
    # a = FFTObject(n)
    # a.data_in[:] = y
    #
    # count = 100000
    # cython_time, python_time = 0, 0
    # for i in range(count):
    # 	start = time.time()
    # 	a.fft()
    # 	end = time.time()
    # 	cython_time += end - start
    #
    # 	start = time.time()
    # 	np.fft.fft(y)
    # 	end = time.time()
    # 	python_time += end - start
    #
    # print("cython", cython_time)
    # print("python", python_time)
    #
    # print("pure cython loop", function())
