// REALLY IMPORTANT TO INCLUDE complex.h before fftw3.h!!!
// forces it to use the standard C complex number definition
#include "fft_stuff.h"
#include<complex.h>
#include<fftw3.h>
#include<stdio.h>
#include<stdlib.h>

void setup(fft_data *data, int N) {
	data->N = N;
	data->data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);
	data->data_out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N * N);

	data->plan_forward = fftw_plan_dft_3d(N, N, N, data->data_in, data->data_out, 1, FFTW_MEASURE);
	data->plan_backward = fftw_plan_dft_3d(N, N, N, data->data_out, data->data_in, -1, FFTW_MEASURE);
}

void finalise(fft_data *data) {
	fftw_destroy_plan(data->plan_forward);
	fftw_destroy_plan(data->plan_backward);
	fftw_free(data->data_in);
	fftw_free(data->data_out);
}

void execute_transform_forward(fft_data *data) {
	// If you only ever apply the FFT plan to
	// the array you created the plan with, you
	// can use the function fftw_execute(plan) here.
	// But sometimes you want to use it on other data which
	// isn't the same but is created in the same way, in
	// which case you use the function fftw_execute_dft
	fftw_execute(data->plan_forward);
}

void execute_transform_backward(fft_data *data) {
	fftw_execute(data->plan_backward);
}



