#Gate denoiser code; 
#Author: Marcus Hughes <hughes.jmb@gmail.com>
#Parallelized using parmap
#Note: this code is for non-commercial use only, see Craig DeForest https://www.boulder.swri.edu/~deforest/

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.ndimage
import time


fft = np.fft.fftn
ifft = np.fft.ifftn
xwidth, ywidth, twidth = 12, 12, 12 # the size of the window in the two spatial and one temporal domains respectively
gamma = 3 # the scaling of the noise threshold. 

(xwidth // 2) - 1, (xwidth // 2) + 2


file_name = '/media/cat/4TB/in_vivo/rafa/yuki/170922_ASAP2f_H151D-T400R_IUE_170730/fov3_flash_2x-005/fov3_flash_2x-005_all.npy'
data = np.load(file_name,mmap_mode='c')

print data.shape
data = data[:1000]
print data.shape
data = np.float64(data)
data = np.swapaxes(data,0,1)
data = np.swapaxes(data,1,2)
print data.shape


plt.figure()#figsize=(15,15))
plt.imshow(np.sqrt(data[:,:,5]), origin='lower', interpolation='None', cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()



# define grid
xstart, xend, xstep = xwidth*2, data.shape[0] - (xwidth*2), 4
ystart, yend, ystep = ywidth*2, data.shape[1] - (ywidth*2), 4
tstart, tend, tstep = twidth, data.shape[2] - twidth, 4

x_ = np.arange(xstart, xend, xstep)
y_ = np.arange(ystart, yend, ystep)
t_ = np.arange(tstart, tend, tstep)
coords = []
for t in t_:
    for y in y_:
        for x in x_:
            coords.append((x,y,t))
            
            
print("There are {} image sections".format(len(coords)))

# equation for a hanning window
hanning_window_3D = lambda x, y, t : (np.power(np.sin((x + 0.5)*np.pi / xwidth), 2.0) * 
                                       np.power(np.sin((y + 0.5) * np.pi / ywidth), 2.0) * 
                                       np.power(np.sin((t + 0.5) * np.pi / twidth), 2.0))

# set up our hanning window array
hanning = np.zeros((xwidth, ywidth, twidth))
for x in range(hanning.shape[0]):
    for y in range(hanning.shape[1]):
        for t in range(hanning.shape[2]):
            hanning[x,y,t] = hanning_window_3D(x,y,t)
            
            
time_index = 0
plt.figure()
plt.imshow(hanning[:,:,time_index])
plt.show()

beta_stack = []
N = 10000

# for N random regions of the image cube we will take the fourier transform 
# and use it to estimate the noise model, beta
for i in np.random.choice(len(coords),N):
    x, y, t = coords[i]

    # get the data
    image_section = data[x-xwidth//2 : x+xwidth//2, 
                         y-ywidth//2 : y+ywidth//2, 
                         t-twidth//2 : t+twidth//2].copy()
    image_section *= hanning # multiply by hanning window
    
    # sum of the square root of the image section
    imbar = np.sum(np.sqrt(image_section))
    
    # take the fourier transform and get the magnitude
    beta_fourier = np.fft.fftshift(fft(image_section))
    beta_fourier_magnitude = np.abs(beta_fourier)
    
    # add this beta term to the stack
    beta_stack.append(beta_fourier_magnitude / imbar)

# from all the sections use the median
beta_approx = np.median(np.stack(beta_stack), axis=0)

percentile = 50
beta_approx = np.percentile(np.stack(beta_stack), percentile, axis=0)

plt.figure()
plt.imshow(beta_approx[:,:,6], origin='lower', interpolation='None', 
          vmin=np.min(beta_approx), vmax=0.5*np.max(beta_approx))

plt.colorbar()
plt.show()


del beta_stack


gated_image = np.zeros_like(data)
times = []

#loop over all image sections

#*******************************************************************************************************
#************************************** PARALLEL FUNCTION **********************************************
#*******************************************************************************************************

def process_parallel(image_section, pars):
    ''' Vars required: hanning, beta_approx, gamma, 
    '''

    # adjust by hanning window
    image_section *= pars.hanning
    imbar = np.sum(np.sqrt(image_section))

    # determine fourier magnitude
    fourier = np.fft.fftshift(fft(image_section))
    fourier_magnitude = np.abs(fourier)
    
    # estimate noise and set the threshold to ignore
    noise = pars.beta_approx * imbar
    threshold = noise * pars.gamma
    
    # any magnitude outside this limit is noise
    gate_filter = np.logical_not(fourier_magnitude < threshold)
    # always preserve the center 3x3 components because they're signal
    gate_filter[5:8, 5:8, 5:8] = True
    final_fourier = fourier * gate_filter
    
    # we can use the wiener filter instead of the gate filter here
    #wiener_filter =  (fourier_magnitude / threshold) / (1 + (fourier_magnitude / threshold))
    #final_fourier = fourier * wiener_filter
    
    # adjust by hanning filter again
    final_image = hanning * np.abs(ifft(np.fft.ifftshift(final_fourier)))

    return final_image

class emptyObject():
    pass

#*******************************************************************************************************
#*********************************** PROCESS DATA ******************************************************
#*******************************************************************************************************
parallel = True
#parallel version of gating code
if parallel:
    #Make empty object to hold parameters to pass to parallel function; also ok to just pass them as params in parmap
    pars = emptyObject()    
    pars.hanning=hanning
    pars.beta_approx=beta_approx
    pars.gamma=gamma
    
    #Make array of image sections to send to parallel function
    image_section_array = []
    for i in range(len(coords)):
        x, y, t = coords[i]
        # get image section copy so as not to manipulate it
        image_section_array.append(data[x-xwidth//2 : x+xwidth//2, 
                             y-ywidth//2 : y+ywidth//2, 
                             t-twidth//2 : t+twidth//2])
    print "length of input (image_section_array): ", len(image_section_array)
                             
    #Run parallel version of gate filter
    import parmap
    final_image_array = parmap.map(process_parallel, image_section_array, pars, processes=16)
    print "length of output (final_section_array): ", len(final_image_array)
    
    
    #Add all processed bits back into the empty gated_image array
    for i in range(len(coords)):
        x, y, t = coords[i]
        gated_image[x-xwidth//2 : x+xwidth//2,
                    y-ywidth//2 : y+ywidth//2, 
                    t-twidth//2 : t+twidth//2] += final_image_array[i]

else:
    #Single core version of the above
    for i in range(len(coords)):
        x, y, t = coords[i]
        if i%10000==0: 
            print "...processing image: ", i
        #start = time.time()

        # get image section copy so as not to manipulate it
        image_section = data[x-xwidth//2 : x+xwidth//2, 
                             y-ywidth//2 : y+ywidth//2, 
                             t-twidth//2 : t+twidth//2].copy()

        # adjust by hanning window
        image_section *= hanning
        imbar = np.sum(np.sqrt(image_section))

        # determine fourier magnitude
        fourier = np.fft.fftshift(fft(image_section))
        fourier_magnitude = np.abs(fourier)
        
        # estimate noise and set the threshold to ignore
        noise = beta_approx * imbar
        threshold = noise * gamma
        
        # any magnitude outside this limit is noise
        gate_filter = np.logical_not(fourier_magnitude < threshold)
        # always preserve the center 3x3 components because they're signal
        gate_filter[5:8, 5:8, 5:8] = True
        final_fourier = fourier * gate_filter
        
        # we can use the wiener filter instead of the gate filter here
        #wiener_filter =  (fourier_magnitude / threshold) / (1 + (fourier_magnitude / threshold))
        #final_fourier = fourier * wiener_filter
        
        # adjust by hanning filter again
        final_image = hanning * np.abs(ifft(np.fft.ifftshift(final_fourier)))

        # add into the final gated image
        gated_image[x-xwidth//2 : x+xwidth//2,
                    y-ywidth//2 : y+ywidth//2, 
                    t-twidth//2 : t+twidth//2] += final_image
        
        #return gated
        #times.append(time.time() - start)
    

fig, axs= plt.subplots(1,2, sharex=True, sharey=True)
axs[0].imshow(data[:,:,50], origin='lower', interpolation='None', cmap='gray')
axs[0].set_axis_off()
axs[0].set_title("Before: Noisy image")

axs[1].imshow(gated_image[:,:,50], origin='lower', interpolation='None', cmap='gray')
axs[1].set_axis_off()
axs[1].set_title("After: Noise gated image")

fig.show()


np.save(file_name[:-4]+"_gated", gated_image)       
