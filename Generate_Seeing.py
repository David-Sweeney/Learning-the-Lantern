"""
An example using Opticspy to fit a set of Zernike terms to some arbitrary phase map.
This can be used for the photonic lantern projects for decomposing seeing into
Zernike terms for subseqent use in a Zernike-basis regression (e.g. NN), or to match
measured performance to theoretical best case scenario.

Opticspy can be downloaded from
https://github.com/Sterncat/opticspy
With docs at
http://opticspy.org

I found that opticspy functions which used mplot3d were flaky, probably a matplotlib
version incompatability. But you don't need opticspy's 3D plotting just for this
task.
"""
from opticspy.zernike import fitting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from hcipy import *
from math import pi
import poppy
import astropy.units as u
from scipy.ndimage import zoom

def create_atmosphere(fried_parameter=2):
	D_tel = 8.2 # meter
	pupil_grid = make_pupil_grid(512, D_tel)
	outer_scale = 20 # meter
	velocity = 10 # meter/sec

	Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
	layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
	return layer

def get_seeing_at_time(wavelengths, atmosphere, time):
	"""
	Returns grid of seeing at specified time.
	wavelengths are in metres
	seeing is returned in radians

	The returned quantity is a np.ndarray of seeings at each wavelength specified.
	"""
	atmosphere.evolve_until(time)
	seeings = []
	for wavelength in wavelengths:
		output_phase = atmosphere.phase_for(wavelength)
		seeings += [np.reshape(output_phase, (512, -1))]
	return np.array(seeings)

def decompose_over_zernikes(seeing, number_of_zernikes):
	# Ok now fit Zernike polynomials
	# WARNING - be careful about which term is which! The first 'n' terms as enumerated here
	# are not necessarily the same as the first 'n' terms used in other code (e.g. poppy)

	num_terms = number_of_zernikes # Number of Zernike terms to fit (does not include piston).
	removepiston = True # Remove the piston term?
	remain2D = False # Show a plot of the residuals (what is left over after the fit)?
	barchart = False # Show a bar chat of fitted polynomial terms?

	fit_list, _ = fitting(seeing, num_terms, removepiston=removepiston,
		remain2D=remain2D, barchart=barchart)

	return fit_list

def apply_null_frame(zernike):
	"""
	Combine the found null solution with the current seeing
	----------
	zernike : numpy array
	 	The Zernikes polynomial coefficients which are to be combined with the
		null solution.
	"""
	return zernike + np.array([0.6739, -0.0156, 0.0101, -0.0056, 0.1226,
									0.0029, -0.0949])/2

def convert_to_noll_order(zernike, num_zernikes):
	"""
	Receives Zernikes in Wyant order with piston
	Converts them to Noll order without piston
	"""
	if num_zernikes == 20:
		noll_ordered_zernike = [zernike[1], zernike[2], zernike[3], zernike[5],
					zernike[4], zernike[7], zernike[6], zernike[10], zernike[9],
					zernike[8], zernike[11], zernike[12], zernike[16], zernike[17],
					zernike[13], zernike[14], zernike[18], zernike[19]]
	elif num_zernikes == 21:
		noll_ordered_zernike = [zernike[1], zernike[2], zernike[3], zernike[5],
					zernike[4], zernike[7], zernike[6], zernike[10], zernike[9],
					zernike[8], zernike[11], zernike[12], zernike[16], zernike[17],
					zernike[13], zernike[14], zernike[18], zernike[19], 0, 0]
	elif num_zernikes == 8:
		noll_ordered_zernike = [zernike[1], zernike[2], zernike[3], zernike[5],
					zernike[4], zernike[7], zernike[6]]
	else:
		raise ValueError(f'Can\'t handle {num_zernikes} Zernikes.')
	return noll_ordered_zernike

def create_seeing_data(wavelengths=np.array([1500e-9,]), fried_parameter=2,
						time=30, num_zernikes=20, nn_range=0.2, null=False,
						filename=None, filepath=None):
	"""
	Create seeing and decompose that seeing over the Zernike polynomials
	----------
	wavelengths : numpy array
		Wavelengths for which to calculate seeing.
	fried_parameter : float
		The value of the Fried Parameter for the atmosphere.
	time : int
		The number of seconds of seeing to generate at 24 fps.
	num_zernikes : int
		The number of Zernikes over which to decompose seeing.
	panic : int
		The range of Zernike term over which the neural network can handle.
	null : bool
		Whether to combine the null frame to the seeing
	filename : str
		File name in which to save seeings and zernikes.
	filepath : str
		File path at which to save seeings and zernikes. If filepath is
		specified then filename is ignored.
	"""
	# Get some phase map to fit to. Here just use a spherical map as a test.
	size = 256
	# phase_map = make_pupil_grid(size)
	# atmosphere = AtmosphericLayer(phase_map, Cn_squared=1)
	# atmosphere = make_standard_atmospheric_layers(phase_map)
	atmosphere_layer = create_atmosphere(fried_parameter)


	# plt.figure()
	# plt.imshow(phase_map)
	# plt.colorbar()
	# plt.title('Input phase map')

	seeings = []
	zernikes = []
	max_zernike_values = []
	RMS_WFEs = []
	times_panicked = 0
	if time <= 30:
		time_in_file = time
	else:
		raise ValueError('Can\'t currently handle more than 30 seconds of seeing in a file.')
		time_in_file = 30
	for file_num in range(int(time / time_in_file)):
		for time in np.arange(file_num*time_in_file, time_in_file + file_num*time_in_file, 1/24):
			seeing = get_seeing_at_time(wavelengths, atmosphere_layer, time)
			if num_zernikes != 21:
				# Convert from radians to microns
				seeing = seeing / (2*pi) * wavelengths.reshape(-1, 1, 1) * 1e6
			set_of_zernikes = []
			for i in range(len(seeing)):
				print(f'Time = {time:.2f}')
				if num_zernikes == 21:
					zernike = decompose_over_zernikes(seeing[i], 20)
					print(f'Mean, largest so far: {np.mean(max_zernike_values)}')
					print(f'Times panicked: {times_panicked}')
				else:
					zernike = decompose_over_zernikes(seeing[i], num_zernikes)
					print(zernike)
					print('!!', num_zernikes)
				zernike = convert_to_noll_order(zernike, num_zernikes)
				if null:
					if num_zernikes != 8:
						raise ValueError('Only support null frame for the 8 Zernike case')
					zernike = apply_null_frame(zernike)
				if len(wavelengths) > 1:
					set_of_zernikes += [np.array(zernike + [wavelengths[i]])]
				else:
					set_of_zernikes += [zernike]
				max_zernike_values += [np.amax(np.abs(zernike))]
				RMS_WFEs +=  [np.std(seeing[i])]
				if np.amax(np.abs(zernike)) > nn_range:
					print('#'*40)
					print('PANIC!')
					print('#'*40)
					times_panicked += 1

			seeings += [seeing]
			zernikes += [set_of_zernikes]
			print('*************')
			print(f'Time = {time:.2f}')
			# print(zernike)
			print('Largest Zernike:', np.amax(np.abs(set_of_zernikes)))
			print(f'Largest so far: {np.amax(max_zernike_values)}')
			print(f'Mean, largest so far: {np.mean(max_zernike_values)}')
			print(f'Times panicked: {times_panicked}')
			print('RMS WFE:', np.std(seeing))
			print('Max RMS WFE:', np.amax(RMS_WFEs))
			print('Mean RMS WFE:', np.mean(RMS_WFEs))
			print('*************')

		print('*************')
		print(f'Max zernike value was {np.amax(max_zernike_values)}')
		print(f'Mean, max zernike value was {np.mean(max_zernike_values)}')
		print('Max RMS WFE:', np.amax(RMS_WFEs))
		print('Mean RMS WFE:', np.mean(RMS_WFEs))
		print('*************')
		# np.savez(f'/media/tintagel/david/opticspy/seeing_19_zernike_1F_{file_num}.npz', seeings=np.array(seeings), zernikes=np.array(zernikes))
		if filepath:
			np.savez(f'{filepath}.npz', seeings=np.array(seeings), zernikes=np.array(zernikes))
		else:
			np.savez(f'/media/tintagel/david/opticspy/{filename}.npz', seeings=np.array(seeings), zernikes=np.array(zernikes))
		# np.savez(f'/import/tintagel3/snert/david/opticspy/seeing_19_zernike_1F_broadband_{file_num}.npz', seeings=np.array(seeings), zernikes=np.array(zernikes))
		seeings = []
		zernikes = []
		max_zernike_values = []
		RMS_WFEs = []

if __name__ == '__main__':
	apply_null_frame(None)
