from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Input, Add, Activation
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as kb
import numpy as np
import pickle
import os
import socket
import traceback
from math import sin, cos, radians, sqrt, pi

# # TODO:
    # ResNet structure can only handle even numbers of layers.
    # ResNet structure can only be accessed through manually changing resnet=True in constructor.
    # augment_data will have to be turned off for reverse models (or changed)
    # augment_data ONLY works for 8 Zernikes---seems to work now but I am unaware of when this changed.

# List of changes for Athena
    # Radians in plot_model units
    # Zernike range

class ModelInfo:
    def __init__(self, datasets, model_name, suffix, model_layers=None, epochs=5000,
            model=None, transfer_datasets=None, learning_rate=0.001, l2_rate=0,
            dropout=None, resnet=False, training_examples=None, test_size=0.02,
            val_size=0.02, augmenting_data=True, broadband_data=False):
        self.model = model
        self.input_data = ''
        self.output_data = ''
        self.datasets = datasets
        self.transfer_datasets = transfer_datasets
        self.model_name = model_name
        self.suffix = suffix     # For saving the model
        self.reversed = suffix[:2] == '-r'
        if model is None:
            if self.reversed:
                self.model_layers = [19] + model_layers + [6] # For reversed photonic lantern
                self.input_data = 'fluxes'
                self.output_data = 'zernikes'
            else:
                self.model_layers = [6] + model_layers + [19] # For photonic lantern emulator
                self.input_data = 'zernikes'
                self.output_data = 'fluxes'
        else:
            if self.reversed:
                self.model_layers = [19] + [0]*14 + [7]
                self.output_data = 'zernikes'
                self.input_data = 'fluxes'
            else:
                self.model_layers = [7] + [0]*14 + [19]
                self.input_data = 'zernikes'
                self.output_data = 'fluxes'
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate
        self.dropout = dropout
        self.dataset_size = None
        self.training_examples = training_examples
        self.augmenting_data = augmenting_data
        self.zernikes = 0
        self.zernike_scaler = None
        self.flux_scaler = None
        self.flux_min = None
        self.average_intensity = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.test_size = test_size
        self.val_size = val_size
        self.early_stopping_patience = 50
        self.resnet = resnet
        self.custom_loss = False
        self.broadband_data = broadband_data

    def rescale_data(self, data, zernike=True):
        """
        All Zernike data should be drawn from a uniform Zernike distribution
        MinMaxScaler is used to transform this distribution to [-1, 1]
        The first data passed to this function sets the fit of the MinMaxScaler
        Transfer data should then be scaled similarly

        If flux data is passed to this function zernike=False
        This data is then normalised to a distribution of [0, 1], then as above.
        """
        if zernike:
            if self.zernike_scaler is None:
                self.zernike_scaler = MinMaxScaler(feature_range=(-1, 1))
                self.zernike_scaler.fit(data)
            data = self.zernike_scaler.transform(data)
        else:
            if self.flux_scaler is None:
                self.flux_scaler = MinMaxScaler(feature_range=(0, 1))
                self.flux_scaler.fit(data)
            data = self.flux_scaler.transform(data)
        return data

    def reverse_data_scaling(self, data, zernike=True):
        """
        Use MinMaxScaler to reverse scaling originally applied
        If the data is the Zernike coefficients, zernike=True otherwise it is
        the fluxes.
        """
        if zernike:
            if self.zernike_scaler is None:
                raise TypeError('Zernike normalisation never took place (scaler was not defined).')
            return self.zernike_scaler.inverse_transform(data)
        else:
            if self.flux_scaler is None:
                raise TypeError('Flux normalisation never took place (scaler was not defined).')
            return self.flux_scaler.inverse_transform(data)

    def read_in_data(self, datasets):
        # Initialise
        zernike_data = None
        flux_data = None
        for i in range(len(datasets)):
            # Read through list
            print('Importing', datasets[i])
            filename = '{}/Neural_Nets/Data/{}/NN_data.npz'.format(get_filepath(), datasets[i])
            npz_file = np.load(filename)
            if zernike_data is None:
                zernike_data = npz_file['zernikes'].copy()
                flux_data = npz_file['fluxes'].copy()
            else:
                zernike_data = np.concatenate((zernike_data, npz_file['zernikes'].copy()))
                flux_data = np.concatenate((flux_data, npz_file['fluxes'].copy()))
        return zernike_data, flux_data

    def normalise_data(self, datasets, zernike_data, flux_data):
        zernike_data = self.rescale_data(zernike_data)

        # Do this for fluxes
        if 'PracticalData' in datasets[0]:
            if self.flux_min is None:
                self.flux_min = np.amin(flux_data)
            flux_data = flux_data - self.flux_min

        if self.reversed:
            flux_data = self.rescale_data(flux_data, zernike=False)
        else:
            if self.average_intensity is None:
                total_intensity = [sum(t) for t in flux_data]
                self.average_intensity = sum(total_intensity)/len(total_intensity)
                print(self.average_intensity)
            flux_data = flux_data / self.average_intensity

        return zernike_data, flux_data

    def rotation_coefficient_generator(self):
        coeffs = None
        j = 1
        m = 1
        sign = 1
        while True:
            for n in reversed(range(m, 0, -2)):
                if m % 2 == 0 and n <= 2:
                    # print(j, m, 0)
                    yield 0
                    j += 1
                    sign *= -1
                # print(j, m, sign*n)
                yield sign*n
                j += 1
                # print(j, m, -sign*n)
                yield -sign*n
                j += 1
            m += 1

    def rotate_data(self, coeffs, fluxes, angle=60):
        rotated_coeffs = np.zeros(len(coeffs))
        look_forward = True
        generator = self.rotation_coefficient_generator()
        # If the data is broadband then the final Zernike coefficient is the wavelength
        for i in range(len(rotated_coeffs)-int(self.broadband_data)):
            rotation_coefficient = next(generator)
            if rotation_coefficient == 0:
                rotated_coeffs[i] = coeffs[i]
                continue
            if look_forward:
                rotated_coeffs[i] = coeffs[i]*cos(rotation_coefficient*radians(angle)) + coeffs[i+1]*sin(rotation_coefficient*radians(angle))
            else:
                rotated_coeffs[i] = coeffs[i]*cos(rotation_coefficient*radians(angle)) + coeffs[i-1]*sin(rotation_coefficient*radians(angle))
            look_forward = not look_forward

        # Rotate fluxes by 60 degrees
        rotated_fluxes = fluxes.copy()
        # Inner hexagon rotates all round by 1
        rotated_fluxes.insert(1, rotated_fluxes.pop(6))
        # Outer hexagon rotates all round by 2
        rotated_fluxes.insert(7, rotated_fluxes.pop(18))
        rotated_fluxes.insert(7, rotated_fluxes.pop(18))

        return rotated_coeffs, rotated_fluxes

    def augment_data(self, zernike_data, flux_data):
        # zernike_data and flux_data are lists
        # Returns augmented versions as NumPy arrays
        augmented_zernikes = []
        augmented_fluxes = []

        for i in range(len(zernike_data)):
            # Include original example
            augmented_zernikes += [zernike_data[i]]
            augmented_fluxes += [flux_data[i]]
            for _ in range(5):
                # Rotate each example 5 times
                rotated_coeffs, rotated_fluxes = self.rotate_data(augmented_zernikes[-1], augmented_fluxes[-1])
                augmented_zernikes += [rotated_coeffs]
                augmented_fluxes += [rotated_fluxes]

        return np.array(augmented_zernikes), np.array(augmented_fluxes)

    def load_data(self, load_specialised=False, original_dataset=None):
        """
        This function takes the datasets in datasets, reads them in and creates
        training and test sets based on the data once it has been shuffled. This
        processess is repeatable since a random_state is used.
        """

        # Pull data in from file and normalise it
        zernike_data, flux_data = self.read_in_data(self.datasets)
        # self.zernike_max = np.amax(zernike_data)

        if load_specialised:
            # Load in the original data set to recreate the normalisation which can then be applied to the specialised data set
            zernike_data_original, flux_data_original = self.read_in_data(original_dataset)
            self.zernike_max = np.amax(zernike_data_original)
            _, _ = self.normalise_data(original_dataset, zernike_data_original, flux_data_original)

        zernike_data, flux_data = self.normalise_data(self.datasets, zernike_data,
                                                        flux_data)

        self.zernikes = len(zernike_data[0])

        if self.transfer_datasets is not None:
            # Pull in any transfer learning data and normalise it
            print('Loading in transfer data')
            transfer_zernike_data, transfer_flux_data = self.read_in_data(self.transfer_datasets)
            transfer_zernike_data, transfer_flux_data = self.normalise_data(self.transfer_datasets, transfer_zernike_data, transfer_flux_data)

        # Assign input/output data
        input_data = []
        output_data = []
        if self.input_data == 'fluxes':
            input_data = flux_data
            output_data = zernike_data
            if self.transfer_datasets is not None:
                transfer_input_data = transfer_flux_data
                transfer_output_data = transfer_zernike_data
        else:
            input_data = zernike_data
            output_data = flux_data
            if self.transfer_datasets is not None:
                transfer_input_data = transfer_zernike_data
                transfer_output_data = transfer_flux_data

        # Split off testing and validation sets
        self.train_x, self.test_x, self.train_y, self.test_y = \
                                    train_test_split(input_data, output_data,
                                    test_size=self.test_size, random_state=0)
        self.train_x, self.val_x, self.train_y, self.val_y = \
                                    train_test_split(self.train_x, self.train_y,
                                    train_size=self.training_examples,
                                    test_size=self.val_size, random_state=0)
        if self.transfer_datasets is not None:
            self.train_x, self.train_y = shuffle(np.concatenate((self.train_x, transfer_input_data)), np.concatenate((self.train_y, transfer_output_data)), random_state=0)

        if self.augmenting_data:
            print('Augmenting training set with rotations')
            self.train_x, self.train_y = self.augment_data(self.train_x.tolist(), self.train_y.tolist())

        # Update number of training examples
        if self.training_examples is None:
            self.training_examples = len(self.train_x)
        self.dataset_size = len(self.train_x) + len(self.val_x) + len(self.test_x)
        return

    def load_pre_split_sets(self, reversed=False):
        """
        Load data in from file directly into training, validation and testing
        sets. Currently only set up to load in data from single file of name
        'NN_data_split.npz'. The data is then normalised and augmented as normal.
        """
        if reversed:
            filename = '{}/Neural_Nets/Data/{}/NN_data_reversed.npz'.format(get_filepath(),
                                                                    self.datasets[0])
        else:
            filename = '{}/Neural_Nets/Data/{}/NN_data_split.npz'.format(get_filepath(),
                                                                    self.datasets[0])
        npz_file = np.load(filename)
        train_x, train_y, val_x, val_y, test_x, \
            test_y = npz_file['train_zernikes'], npz_file['train_fluxes'], \
            npz_file['validation_zernikes'], npz_file['validation_fluxes'], \
            npz_file['test_zernikes'], npz_file['test_fluxes']

        train_x, train_y = self.normalise_data(self.datasets, train_x, train_y)
        val_x, val_y = self.normalise_data(self.datasets, val_x, val_y)
        test_x, test_y = self.normalise_data(self.datasets, test_x, test_y)

        self.zernikes = len(train_x[0])

        if self.input_data == 'fluxes':
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, \
                self.test_y = train_y, train_x, val_y, val_x, test_y, test_x
        else:
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, \
                self.test_y = train_x, train_y, val_x, val_y, test_x, test_y

        # Update number of training examples
        if self.training_examples is None:
            self.training_examples = len(self.train_x)
        else:
            self.train_x = self.train_x[:self.training_examples]
            self.train_y = self.train_y[:self.training_examples]

        if self.augmenting_data:
            print('Augmenting training set with rotations')
            self.train_x, self.train_y = self.augment_data(self.train_x.tolist(),
                                                            self.train_y.tolist())
        print('Set sizes:', len(self.train_x), len(self.val_x), len(self.test_x))
        self.dataset_size = len(self.train_x) + len(self.val_x) + len(self.test_x)
        return

    def plot_training(self, model, history, test_x, test_y, predicted_y, filepath, show_plots):#, model_name='model', suffix=''):
        # To plot:
        # Loss
        # Metric
        # Model infrastructure (3)
        # Hyperparameters
        # Random selection of test data

        # Metric
        plt.figure(figsize=(20, 10))
        plt.subplot(3, 3, 3)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.legend(['Training Set', 'Validation Set'], loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Mean Absolute Error')
        # print(history.history.keys())
        plt.ylim([0, np.percentile(history.history['val_mae'], 97)])
        # plt.show()

        # Loss
        plt.subplot(3, 3, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        if self.custom_loss:
            plt.ylabel('Weighted Mean Squared Error')
        else:
            plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.legend(['Training Set', 'Validation Set'], loc='upper right')
        plt.ylim([0, np.percentile(history.history['val_loss'], 97)])
        # plt.show()

        # Model infrastructure
        plt.subplot(1, 3, 1)
        # plotted_model = K.utils.model_to_dot(model, show_shapes=True, show_layer_names=True)
        K.utils.plot_model(model, to_file=filepath + 'model.png', show_shapes=True, show_layer_names=True, dpi=2500) # dpi=800/(len(self.model_layers)*3-4)
        # K.utils.model_to_dot(model, show_shapes=True, show_layer_names=True).write_png(filepath + 'model.png')
        model_image = mpimg.imread(filepath + 'model.png')
        plt.imshow(model_image, aspect='equal')
        plt.axis('off')
        plt.title('{}{} Structure'.format(self.model_name, self.suffix))

        # Hyperparameters
        plt.subplot(3, 3, 8)
        plt.axis('off')
        plt.title('Hyperparameters')
        if self.custom_loss:
            loss = '{}\n{}Weighted MSE: {}'.format(np.amin(history.history['val_mse']), ' '*10, np.amin(history.history['val_loss']))
        elif 'weighted_mse_loss' in history.history.keys():
            loss = '{}\n{}Weighted MSE: {}'.format(np.amin(history.history['val_loss']), ' '*10, np.amin(history.history['val_weighted_mse_loss']))
        else:
            loss = '{}'.format(np.amin(history.history['val_loss']))
        layers = self.model_layers
        if len(layers) >= 10:
            # The text will warp the plotting
            layers = '{} layers with an average hidden layer size of {}'.format(len(layers)-1, sum(layers[1:-1])/(len(layers)-2))
        plt.text(0, 0, 'Model Name: {}\nZernike Modes: {}\nSuffix: {}\nDatasets: \
        {}\nTransfer Datasets: {}\nTotal Dataset size: {}\nTraining Examples: {}\n\
        Model Layers: {}\nActivation Function: {}\nLearning Rate: {}\nDropout: {}\
        \nLambda (L2_rate): {}\nBatch Normalisation: {}\nEarly Stopping: {} epochs\
        \nReduce LR on Plateau: {}\nBest Validation MSE: {}\n{} MAE: {}\n'.format(
            self.model_name, self.zernikes, self.suffix, self.datasets,
            self.transfer_datasets, self.dataset_size, self.training_examples,
            layers, 'LeakyReLU', self.learning_rate, self.dropout, self.l2_rate,
            'On',  self.early_stopping_patience, 'On', loss, ' '*24,
            np.amin(history.history['val_mae'])))

        # 3 random results from the test set plotted (5, 6, 9)
        plot_locations = (5, 6, 9)
        x_axis = np.arange(1, len(test_y[0])+1)
        y_max = max(np.amax(predicted_y), np.amax(test_y)) + 0.05
        y_min = min(np.amin(predicted_y), np.amin(test_y)) - 0.05
        for i in range(3):
            plt.subplot(3, 3, plot_locations[i])
            plt.plot(x_axis, [0]*len(x_axis), 'k--', linewidth=0.5)
            plt.scatter(x_axis, predicted_y[i], marker='x', alpha=0.7, label='Predicted Value')
            plt.scatter(x_axis, test_y[i], marker='.', alpha=0.7, label='True Value')
            plt.scatter(x_axis, predicted_y[i] - test_y[i], marker='s', alpha=0.6, label='Difference')
            plt.legend(loc='lower right')
            if len(test_x[i]) == 21 or (self.reversed and 'Spectrum' in self.datasets[0]):
                plt.title(f'Random Test Data, $\lambda = {test_x[i][-1]:.0f}nm$')
            else:
                plt.title('Random Test Data')
            if len(x_axis) > 17:
                plt.xticks(x_axis, [i if i % 2 == 1 else '' for i in x_axis])

            if self.reversed:
                plt.ylim([min(y_min, -0.35), max(y_max, 0.35)])
                plt.xlim([0.75, 20.25])
                plt.xlabel('Zernike Mode Number')
                if 'PracticalData' in self.datasets[0]:
                    plt.ylabel('Zernike Value (radians)')
                else:
                    plt.ylabel('Zernike Value (microns)')
            else:
                plt.ylim([-0.2, max(y_max, 0.2)])
                plt.xlim([0.75, 19.25])
                plt.xlabel('Waveguide Number')
                plt.ylabel('Flux Value (arb. units)')

        plt.tight_layout(pad=1.2)
        # plt.savefig(filepath+'{}{}.png'.format(self.model_name, self.suffix), dpi=1000)
        plt.savefig(filepath+'{}{}.png'.format(self.model_name, self.suffix))
        if show_plots:
            plt.show()
        plt.close()

    def plot_training_examples(self, model, train_x, train_y, predicted_y, filepath):#, model_name='model', suffix=''):
        # To plot:
        # Loss
        # Metric
        # Model infrastructure (3)
        # Hyperparameters
        # Random selection of test data

        # Metric
        plt.figure(figsize=(20, 10))
        # 3 random results from the test set plotted (5, 6, 9)
        x_axis = np.arange(1, len(train_y[0])+1)
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.plot(x_axis, [0]*len(x_axis), 'k--', linewidth=0.5)
            plt.scatter(x_axis, predicted_y[i], marker='x', alpha=0.7, label='Predicted Value')
            plt.scatter(x_axis, train_y[i], marker='.', alpha=0.7, label='True Value')
            plt.scatter(x_axis, predicted_y[i] - train_y[i], marker='s', alpha=0.6, label='Difference')
            plt.legend(loc='lower right')
            plt.title('Random Training Data')
            plt.ylim([-0.5, 0.5])
            if len(train_y[i]) == 19:
                plt.xlabel('Waveguide Number')
                plt.ylabel('Flux Value (arb. units)')
            else:
                plt.xlabel('Zernike Mode Number')
                plt.ylabel('Zernike Value (microns)')

        plt.tight_layout(pad=1.2)
        plt.savefig(filepath+'{}{}_training.png'.format(self.model_name, self.suffix))
        plt.close()

    def weighted_mse_loss(self, y_actual, y_pred):
        squared = kb.square(y_actual-y_pred)
        weighted_mse = (kb.sum(squared, axis=1) + 99*squared[:, 0])/19
        return weighted_mse

    def run(self, show_plots=True, show_training_predictions=False, verbose=1, load=True, load_sets=False):
        print('Loading in data')
        if load_sets:
            self.load_pre_split_sets(reversed=self.reversed)
        elif load:
            self.load_data()

        # Fix layer sizes for number of Zernikes
        if self.reversed:
            self.model_layers[-1] = self.zernikes
            self.model_layers[0] = self.train_x.shape[-1]
        else:
            self.model_layers[0] = self.zernikes

        # Create model
        if self.resnet:
            print('Creating ResNet style model')
            model = self.create_model_resnet()
        else:
            print('Creating model')
            model = self.create_model()

        # Compile model
        if self.custom_loss:
            model.compile(optimizer=K.optimizers.Adam(learning_rate=self.learning_rate), loss=self.weighted_mse_loss, metrics=['mse', 'mae'])
        else:
            # model.compile(optimizer=K.optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae', self.weighted_mse_loss])
            model.compile(optimizer=K.optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae',])

        # Create callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, verbose=0, min_delta=5e-4, mode='min')
        # mcp_save = ModelCheckpoint('{}/Neural_Nets/Models/{}/{}_model{}.h5'.format(get_filepath(), self.model_name, self.model_name, self.suffix), save_only_best=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, verbose=1, min_delta=5e-4, mode='min')

        print('Training model')
        history = model.fit(self.train_x, self.train_y, validation_data=(self.val_x, self.val_y), epochs=self.epochs, callbacks=[early_stopping, reduce_lr_loss], verbose=verbose, batch_size=32)
        # history2 = model.fit(self)
        # history = model.fit(train_images, train_labels, validation_split=0.25, epochs=1000, batch_size=32, workers=16, use_multiprocessing=True)

        print('Plotting model')
        save_path = '{}/Neural_Nets/Models/{}/'.format(get_filepath(), self.model_name)
        try:
            test_x, test_y, predicted_y = self.test_n_data(model, self.test_x, self.test_y, 3)
            self.plot_training(model, history, test_x, test_y, predicted_y, save_path, show_plots) #, model_name=self.model_name, suffix=self.suffix)
        except Exception as e:
            print('***********************************************\nFailed to plot model\n')
            traceback.print_exc()
            print()

        if show_training_predictions:
            train_x, train_y, predicted_y = self.test_n_data(model, self.train_x, self.train_y, 9)
            self.plot_training_examples(model, train_x, train_y, predicted_y, save_path)
        print('Saving model')
        save_model(model, save_path + '{}_model{}.h5'.format(self.model_name, self.suffix), history=history)
        # model.summary()
        # print('Saving details')
        # save_details(self.model_name, self.input_data, self.output_data, self.datasets, self.suffix, self.model_layers, history, save_path)
        return history

    def train_specialised_model(self, original_dataset, show_plots=False, show_training_predictions=True, verbose=0):
        print('Loading in data')
        self.load_data(load_specialised=True, original_dataset=original_dataset)

        kb.set_value(self.model.optimizer.learning_rate, self.learning_rate)

        # Create callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, verbose=0, min_delta=1e-8, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, verbose=1, min_delta=1e-8, mode='min')

        print('Training model')
        history = self.model.fit(self.train_x, self.train_y, validation_data=(self.val_x, self.val_y), epochs=self.epochs, callbacks=[early_stopping, reduce_lr_loss], verbose=verbose, batch_size=32)

        print('Plotting model')
        save_path = '{}/Neural_Nets/Models/{}/'.format(get_filepath(), self.model_name)
        try:
            val_x, val_y, predicted_y = self.test_n_data(self.model, self.val_x, self.val_y, 3)
            self.plot_training(self.model, history, val_x, val_y, predicted_y, save_path, show_plots) #, model_name=self.model_name, suffix=self.suffix)
        except Exception as e:
            print('***********************************************\nFailed to plot model\n')
            traceback.print_exc()
            print()

        if show_training_predictions:
            train_x, train_y, predicted_y = self.test_n_data(self.model, self.train_x, self.train_y, 9)
            self.plot_training_examples(self.model, train_x, train_y, predicted_y, save_path)
        print('Saving model')
        save_model(self.model, save_path + '{}_model{}.h5'.format(self.model_name, self.suffix), history=history)
        # model.summary()
        # print('Saving details')
        # save_details(self.model_name, self.input_data, self.output_data, self.datasets, self.suffix, self.model_layers, history, save_path)
        return history

    def test_n_data(self, model, set_x, set_y, num):
        # ONLY SET UP FOR OUTPUTTING FLUXES AT THE MOMENT
        # Randomly select num instances from test data to predict
        # Stores random choices in an array of length num
        new_set_x, _, new_set_y, _ = train_test_split(set_x, set_y, train_size=num)
        predicted_y = model.predict(new_set_x)
        if self.reversed:
            # We are then in a reverse model, predicting Zernikes
            # So we should undo the normalisation
            predicted_y = self.reverse_data_scaling(predicted_y)
            new_set_y = self.reverse_data_scaling(new_set_y)
            new_set_x = self.reverse_data_scaling(new_set_x, zernike=False)

        else:
            # We are in a forward model, so need to undo the normalisation on
            # the x data
            new_set_x = self.reverse_data_scaling(new_set_x)
        return new_set_x, new_set_y, predicted_y

    def normal_block(self, layer_size, input):
        x = Dense(layer_size, kernel_regularizer=regularizers.l2(self.l2_rate))(input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        # x = tanh(x)
        # x = sigmoid(x)
        if self.dropout is not None:
            x = Dropout(self.dropout)(x)
        return x

    def resnet_block(self, layer_size, input):
        x = Dense(layer_size[0], kernel_regularizer=regularizers.l2(self.l2_rate))(input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.dropout is not None:
            x = Dropout(self.dropout)(x)

        # Second block
        x = Dense(layer_size[1], kernel_regularizer=regularizers.l2(self.l2_rate))(x)

        # Recombine
        x = Add()([input, x])
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.dropout is not None:
            x = Dropout(self.dropout)(x)
        return x

    def create_model_resnet(self):
        x_input = Input(self.model_layers[0])
        x = Dense(self.model_layers[1], kernel_regularizer=regularizers.l2(self.l2_rate), name = 'First_Hidden_Layer')(x_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(2, len(self.model_layers)-2, 2):
            x = self.resnet_block(self.model_layers[i:i+2], x)
            # # Split off shortcut path
            # x_shortcut = x
            # # First main block
            # x = Dense(self.model_layers[i], kernel_regularizer=regularizers.l2(self.l2_rate))(x)
            # x = LeakyReLU()(x)
            # x = BatchNormalization()(x)
            # if self.dropout is not None:
            #     x = Dropout(self.dropout)(x)
            #
            # # Second block
            # x = Dense(self.model_layers[i+1], kernel_regularizer=regularizers.l2(self.l2_rate))(x)
            #
            # # Recombine
            # x = Add()([x_shortcut, x])
            # x = LeakyReLU()(x)
            # x = BatchNormalization()(x)
            # if self.dropout is not None:
            #     x = Dropout(self.dropout)(x)

        x = Dense(self.model_layers[-2], kernel_regularizer=regularizers.l2(self.l2_rate), name = 'Final_Hidden_Layer')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dense(self.model_layers[-1])(x)
        model = K.Model(inputs = x_input, outputs = x, name = self.model_name)
        return model

    def create_model(self):
        x_input = Input(self.model_layers[0])
        # x = self.normal_block(self.model_layers[1], x_input)
        x = Dense(self.model_layers[1], kernel_regularizer=regularizers.l2(self.l2_rate))(x_input)#, name = 'First_Hidden_Layer')(x_input)
        x = LeakyReLU()(x)
        # x = tanh(x)
        # x = sigmoid(x)
        x = BatchNormalization()(x)
        for i in range(1, len(self.model_layers)-2):
            x = self.normal_block(self.model_layers[i], x)
            # x = Dense(self.model_layers[i], kernel_regularizer=regularizers.l2(self.l2_rate))(x)
            # x = LeakyReLU()(x)
            # x = BatchNormalization()(x)
            # if self.dropout is not None:
            #     x = Dropout(self.dropout)(x)
        x = Dense(self.model_layers[-1], kernel_regularizer=regularizers.l2(self.l2_rate))(x)
        model = K.Model(inputs = x_input, outputs = x, name = self.model_name)
        # model = K.Sequential()
        # for i in range(len(model_layers)):
        #     if i == 0 or i == len(model_layers) - 1:
        #         model.add(Dense(model_layers[i]))
        #     else:
        #         model.add(Dense(model_layers[i], kernel_regularizer=regularizers.l2(l2_rate)))
        #         model.add(LeakyReLU(alpha=0.01))
        #         model.add(BatchNormalization())
        #         if dropout is not None:
        #             model.add(Dropout(dropout))
        return model

def get_filepath():
    computer_name = socket.gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    else:
        return '/import/tintagel3/snert/david'

def test_model(model, test_x, test_y):
    # Run model on testing set
    test_result = model.evaluate(test_images, test_labels)
    return test_result

def save_model(model, filepath, history=None, history_path=None):
    if history is not None:
        # Save history
        try:
            if history_path is None:
                history_path = filepath[:-3] + '_history.npz'
            # with open(history_path, 'w') as file:
                # pickle.dump(history, file, pickle.HIGHEST_PROTOCOL)
            np.savez(history_path, loss=history.history['loss'], val_loss=history.history['val_loss'], mae=history.history['mae'], mae_loss=history.history['val_mae'])
        except Exception:
            print('***********************************************\nCould not save history:\n')
            print(traceback.print_exc())
            print()

    # Save model
    model.save(filepath)

def load_model(filepath, load_history=False, history_path=None):
    # Will return 2 objects if load_history=True
    model = K.models.load_model(filepath)
    try:
        if load_history and history_path is not None:
            history = 0
            with open(history_path, 'r') as file:
                history = pickle.load(file)
            return model, history
    except:
        print('***********************************************\nCould not load history\n')
        traceback.print_exc()
        print()
    return model

def save_details(model_name, input_data, output_data, datasets, suffix, model_layers, history, save_path):
    filename = save_path + '{}{}_info.txt'.format(model_name, suffix)
    # If file already exists, append something to name to create new file
    while os.path.isfile(filename):
        filename = filename[:-4] + '1.txt'
    # Save run related info
    with open(filename, 'w') as file:
        file.write('input_data={}\n'.format(input_data))
        file.write('output_data={}\n'.format(output_data))
        file.write('datasets={}\n'.format(datasets))
        file.write('model_name={}\n'.format(model_name))
        file.write('suffix={}\n'.format(suffix))
        file.write('model_layers={}\n'.format(model_layers))
        file.write('Best Validation MSE={}, MAE={}\n'.format(np.percentile(history.history['val_loss'], 2), np.percentile(history.history['val_mae'], 2)))

def save_history(filepath, history=None):
    if history is not None:
        # Save history
        try:
            with open(filepath, 'w') as file:
                pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)
        except:
            print('***********************************************\nCould not save history')
            traceback.print_exc()
            print()

def weighted_mse_loss(y_actual, y_pred):
    squared = kb.square(y_actual-y_pred)
    weighted_mse = (kb.sum(squared, axis=1) + 99*squared[:, 0])/19
    return weighted_mse

def plot_best_model_test_predictions():
    font = {'size'   : 18}
    matplotlib.rc('font', **font)
    # for model_full_name in ['Simple-b57', 'Hestia5-f50', 'Hyperion5-d3', 'Apollo-b1', 'Orpheus-d3', 'Themis-3-b27', 'Athena-t1']:
    # for model_full_name in ['Simple-b57', 'Apollo-b1', 'Themis-3-b27', 'Athena-b58',]:
    # for model_full_name in ['Orpheus-d3',]:
    # for model_full_name in ['Apollo-b1',]:
    for model_full_name in ['Athena-t2',]:
    # for model_full_name in ['Simple-r-b40', 'Apollo-r-d2', 'Orpheus-r']:
    # for model_full_name in ['Orpheus-r',]:
    # for model_full_name in ['Athena-r-t1',]:
        if model_full_name == 'Themis-3-b27':
            model_name, model_suffix = 'Themis-3', 'b27'
        elif model_full_name == 'Orpheus-r':
            model_name, model_suffix = 'Orpheus', 'r'
        elif '-r' in model_full_name:
            model_name, _, model_suffix = model_full_name.split('-')
            model_suffix = 'r-' + model_suffix
        else:
            model_name, model_suffix = model_full_name.split('-')
        filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name, f'{model_name}_model-{model_suffix}')
        if model_name == 'Hyperion5':
            datasets = ('BigOne35',)
        elif model_name == 'Apollo':
            datasets = ('BigOne3', 'BigOne4')
        elif model_name == 'Simple':
            datasets = ('BigOne1', 'BigOne2')
        elif model_name == 'Hestia5':
            datasets = ('BigOne23',)
        elif model_name == 'Orpheus':
            datasets = ('BigOne5', 'BigOne6')
        elif model_name == 'Themis-3':
            datasets = ('BigOne9', 'BigOne14', 'BigOne15')
        elif model_name == 'Athena':
            datasets = ('PracticalData9',)
        print(filepath)
        model = K.models.load_model(filepath, custom_objects={"weighted_mse_loss":weighted_mse_loss})
        print(filepath)
        model_info = ModelInfo(
            datasets = datasets,
            model_name = 'Load',
            suffix = '-l1',
            model = model,
            augmenting_data = False
        )
        model_info.load_data()
        # model_info.model_layers[-1] = model_info.zernikes
        number_of_plots = 32
        rows = 1
        test_x, test_y, predicted_y = model_info.test_n_data(model, model_info.test_x, model_info.test_y, number_of_plots)

        # plt.figure(figsize=(18, 4.5))

        # x_axis = np.arange(1, len(test_y[0])+1)
        # plt.suptitle(model_full_name)
        # plt.subplots_adjust(top=0.75, bottom=0.25)

        # for i in range(number_of_plots):
        #     plt.subplot(rows, int(number_of_plots/rows), i+1)
        #     plt.plot(x_axis, [0]*len(x_axis), 'k--', linewidth=0.5)
        #     plt.scatter(x_axis, predicted_y[i], marker='x', alpha=0.7, label='Predicted Value', s=100)
        #     plt.scatter(x_axis, test_y[i], marker='.', alpha=0.7, label='True Value', s=100)
        #     plt.scatter(x_axis, (predicted_y[i] - test_y[i]), marker='_', alpha=0.9, label='Difference')
        #     if len(x_axis) > 17:
        #         plt.xticks(x_axis, [i if i % 2 == 1 else '' for i in x_axis])
        #     elif len(x_axis) > 14:
        #         plt.xticks(x_axis, [i if i % 2 == 1 else '' for i in x_axis])
        #     else:
        #         plt.xticks(x_axis, x_axis)
        #     # plt.legend(loc='lower right')
        #     # plt.legend()
        #     # plt.xlim([0.75, 19.25])
        #
        #     # Forward model code
        #     plt.ylim([-0.1, 0.40])
        #     plt.title(f'Example {i+1}')
        #     if i % (number_of_plots / rows) == 0:
        #         plt.ylabel('Intensity (arb. units)')
        #     if i >= number_of_plots - 4:
        #         plt.xlabel('Waveguide number')
        #     plt.xlim([0.75, len(test_y[0])+0.25])
        #
        #     # # Reverse model code
        #     # plt.ylim([-0.4, 0.4])
        #     # plt.title(f'Example {i+1}')
        #     # if i % (number_of_plots / rows) == 0:
        #     #     plt.ylabel('Zernike RMS coefficient ($\mu m$)')
        #     # if i >= number_of_plots - (number_of_plots / rows):
        #     #     plt.xlabel('Zernike polynomial')
        #     # plt.xlim([0.75, len(test_y[0])+0.25])
        #
        #
        #     # if len(test_y[i]) == 19:
        #     #     plt.xlabel('Waveguide Number')
        #     #     if i == 0:
        #     #         plt.ylabel('Flux Value (arb. units)')
        #     # else:
        #     #     plt.xlabel('Zernike Mode Number')
        #     #     plt.ylabel('Zernike Value (microns)')

        plt.figure(figsize=(7.56, 7.35))
        plt.plot([0, 0.4], [0, 0.4], 'k--', alpha=0.5)
        plt.scatter(test_y, predicted_y, alpha=0.4, color='tab:blue')
        # plt.scatter(test_y, (test_y - predicted_y), alpha=0.7, color='tab:orange')
        axis_labels = [f'{i:.2f}' for i in np.linspace(0, 0.2, 5)]
        plt.xticks(list(np.linspace(0, 0.2, 5)), axis_labels)
        plt.yticks(list(np.linspace(0, 0.2, 5)), axis_labels)
        plt.ylabel('Predicted value')
        plt.xlabel('True value')
        plt.xlim([0, 0.2])
        plt.ylim([0, 0.2])
        plt.title('16-Zernike Reverse Emulator over $\pm0.2 \mu m$')
        plt.tight_layout(pad=0)
        # plt.savefig(filepath+'{}{}.png'.format(self.model_name, self.suffix), dpi=1000)
        plt.savefig(f'{get_filepath()}/Honours_Report/Model_Test_Set_Figures/{model_full_name}_10.png')
        # plt.show()
        plt.close()

def assess_on_test_set():
    """
    Calculates MSE of model on specified testing set which was trained independently.
    Then plots the correlation plot of this testing set.

    for loop with model_full_name must be changed if fewer models are required.
    """
    font = {'size'   : 18}
    matplotlib.rc('font', **font)
    for model_full_name in ['Simple-b57', 'Apollo-b1', 'Orpheus-d3', 'Themis-3-b27', 'Hestia5-f50', 'Apollo5-b11', 'Hestia10-b18', 'Athena-t9', 'Gaia-na-d3']:
    # for model_full_name in ['Simple-b57', 'Apollo-b1', 'Orpheus-d3', 'Themis-3-b27', 'Hestia5-f50', 'Apollo5-b11', 'Hestia10-b18']:
    # for model_full_name in ['Athena-t9', 'Gaia-na-d3']:
    # for model_full_name in ['Athena-t9',]:
        print('*'*20)
        if model_full_name == 'Themis-3-b27':
            model_name, model_suffix = 'Themis-3', 'b27'
        elif model_full_name == 'Gaia-na-d3':
            model_name, model_suffix = 'Gaia', 'na-d3'
        elif model_full_name == 'Orpheus-r':
            model_name, model_suffix = 'Orpheus', 'r'
        elif '-r' in model_full_name:
            model_name, _, model_suffix = model_full_name.split('-')
            model_suffix = 'r-' + model_suffix
        else:
            model_name, model_suffix = model_full_name.split('-')
        filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name, f'{model_name}_model-{model_suffix}')
        if model_name == 'Simple':
            datasets = ('BigOne1', 'BigOne2')
            test_set = ('Test_Simple48',)
            rms = 0.277
        elif model_name == 'Hestia5':
            datasets = ('BigOne23',)
            test_set = ('Test_Hestia5-52',)
            rms = 0.752
        elif model_name == 'Hestia10':
            datasets = ('BigOne32', 'BigOne34')
            test_set = ('Test_Hestia10-54',)
            rms = 1.50
        elif model_name == 'Apollo':
            datasets = ('BigOne3', 'BigOne4')
            test_set = ('Test_Apollo49',)
            rms = 0.362
        elif model_name == 'Apollo5':
            datasets = ('BigOne8',)
            test_set = ('Test_Apollo5-53',)
            rms = 0.901
        elif model_name == 'Orpheus':
            datasets = ('BigOne5', 'BigOne6')
            test_set = ('Test_Orpheus50',)
            rms = 0.443
        elif model_name == 'Themis-3':
            datasets = ('BigOne9', 'BigOne14', 'BigOne15')
            test_set = ('Test_Themis51',)
            rms = 0.487
        elif model_name == 'Athena':
            datasets = ('PracticalData9',)
            rms = 0.744
        elif model_name == 'Gaia':
            datasets = ('PracticalData19_Spectrum',)
            rms = 0.950
        print(filepath)
        model = K.models.load_model(filepath, custom_objects={"weighted_mse_loss":weighted_mse_loss})
        model_info = ModelInfo(
            datasets = datasets,
            model_name = 'Load',
            suffix = '-l1',
            model = model,
            augmenting_data = False,
            test_size = 0.02,
            val_size = 0.02
        )
        if 'Gaia' in model_name:
            model_info.load_pre_split_sets()
        else:
            model_info.load_data()
        if 'PracticalData' not in datasets[0]:
            raw_test_zernikes, raw_test_fluxes = model_info.read_in_data(test_set)
            test_zernikes, test_fluxes = model_info.normalise_data(test_set, raw_test_zernikes, raw_test_fluxes)
            model_info.test_x = test_zernikes
            model_info.test_y = test_fluxes
        metrics = model.evaluate(model_info.test_x, model_info.test_y, verbose=False)
        print(model_name)
        print(metrics)
        print(model.metrics_names)

        # Correlation plot
        number_of_predictions = 200
        # max_zernike = round(np.amax(raw_test_zernikes), 1)
        max_zernike = 0.38
        max_flux = 0.3
        ticks = 4
        print(f'Max flux: {np.amax(model_info.test_y)}')
        test_x, test_y, predicted_y = model_info.test_n_data(model, model_info.test_x, model_info.test_y, number_of_predictions)
        plt.figure(figsize=(7.56, 7.35))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.scatter(test_y, predicted_y, alpha=0.05, color='tab:blue')
        axis_labels = [f'{i:.1f}' for i in np.linspace(0, max_flux, ticks)]
        plt.xticks(list(np.linspace(0, max_flux, ticks)), axis_labels)
        plt.yticks(list(np.linspace(0, max_flux, ticks)), axis_labels)
        plt.ylabel('Predicted waveguide output intensity (norm. units)')
        plt.xlabel('True waveguide output intensity (norm. units)')
        plt.xlim([0, max_flux])
        plt.ylim([0, max_flux])
        if 'PracticalData' in datasets[0]:
            if 'Gaia' in model_name:
                plt.title(f'20-Zernike Model with ${rms}$ $rad.$ RMS WFE')
            else:
                plt.title(f'{model_info.zernikes+1}-Zernike Model with ${rms}$ $rad.$ RMS WFE')
        else:
            plt.title(f'{model_info.zernikes+1}-Zernike Model with ${rms} \mu m$ RMS WFE')
        plt.tight_layout(pad=0)
        plt.savefig(f'{get_filepath()}/Papers/Learning_The_Lantern/Correlation_Plots/{model_full_name}.png')
        plt.close()
        print('*'*20)

def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    pass

def specialise_model(dataset, model_name, suffix, original_dataset):
    model_name, model_suffix = model_name.split('-')
    filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name, f'{model_name}_model-{model_suffix}')
    model = K.models.load_model(filepath)
    ModelInfo(
        datasets = (dataset, ),
        model_name = model_name,
        suffix = '-{}-sp{}'.format(model_suffix, suffix),
        model = model,
        augmenting_data = False,
        val_size = 0.20,
        test_size = 0.0000000000001,
        learning_rate = 1e-10).train_specialised_model(original_dataset)
    pass

def train_both(datasets, model_name, suffix, model_layers, epochs, learning_rate, l2_rate, dropout=None, training_examples=None, show_plots=True, verbose=1, resnet=False):
    forward = ModelInfo(
        datasets=datasets,
        model_name=model_name,
        suffix = suffix,
        model_layers = model_layers,
        epochs = epochs,
        learning_rate = learning_rate,
        l2_rate = l2_rate,
        dropout = dropout,
        training_examples = training_examples,
        resnet = resnet)
    backward = ModelInfo(
        datasets=datasets,
        model_name=model_name,
        suffix = '-r'+suffix,
        model_layers = model_layers,
        epochs = epochs,
        learning_rate = learning_rate,
        l2_rate = l2_rate,
        dropout = dropout,
        training_examples = training_examples,
        resnet = resnet)
    return forward.run(show_plots=show_plots, verbose=verbose), backward.run(show_plots=show_plots, verbose=verbose)

def train(datasets, model_name, suffix_starts, methods, boths, transfer_datasets=None):
    # Declare the default values which almost all models will use
    default_model_layers = [2000, 2000, 2000]
    default_epochs = 5000
    default_learning_rate = 0.0001
    default_l2_rate = 0 #1*10**-6.8
    for method_num in range(len((methods))):
        # Establish the appropriate values which were passed to the function
        method = methods[method_num]
        suffix_start = suffix_starts[method_num]
        both = boths[method_num]
        if 'regularisation' == method:
            pre_suffix = '-na-d'
            default_model_layers = [2000]*7
            # Regularisation scan from 0->0.5 dropout and e-5->e-8 lambda
            # l2_rates = [1*10**-i for i in np.linspace(5, 8, 16)]
            l2_rates = []
            # dropouts = np.linspace(0, 0.5, 11)
            dropouts = np.linspace(0.1, 0.3, 5)
            regularisations = []
            for i in range(len(l2_rates)):
                regularisations += [(l2_rates[i], 0)]
            for i in range(len(dropouts)):
                regularisations += [(0, dropouts[i])]
            suffixes = [pre_suffix + str(i) for i in range(suffix_start, suffix_start + len(regularisations))]
            for i in range(len(suffixes)):
                print('Starting model with regularisation hyperparameters:', regularisations[i])
                if both:
                    train_both(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = regularisations[i][0],
                        dropout = regularisations[i][1],
                        show_plots = 0,
                        verbose = 0)
                else:
                    ModelInfo(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = regularisations[i][0],
                        dropout = regularisations[i][1],
                        resnet = False,
                        augmenting_data = False,
                        test_size = 0.01,
                        val_size = 0.01,
                        broadband_data = True,
                        training_examples = None).run(show_plots = 0, verbose = 0,
                                            show_training_predictions = False,
                                            load_sets = True)
        elif 'learning rate' == method:
            # Regularisation scan from 0->0.5 dropout and e-5->e-8 lambda
            pre_suffix = '-l'
            learning_rates = [1*10**-i for i in np.linspace(3, 6, 7)]
            suffixes = [pre_suffix + str(i) for i in range(suffix_start, suffix_start + len(learning_rates))]
            for i in range(len(suffixes)):
                print('Starting model with learning rate:', learning_rates[i])
                if both:
                    train_both(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = learning_rates[i],
                        l2_rate = default_l2_rate,
                        show_plots = 0,
                        verbose = 0)
                else:
                    ModelInfo(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = learning_rates[i],
                        l2_rate = default_l2_rate).run(show_plots = 0, verbose = 0)
        elif 'stubby sizes' == method:
            # Explore some pre-defined layer sizes
            pre_suffix = '-b'
            layers = [[500]*8, [2000]*6, [2000]*5, [2000]*4, [2000]*3, [2000]*2, [1000], [500]]
            # layers = [[500]*7, [2000, 2000, 2000, 2000, 2000, 2000], [2000, 2000, 2000, 2000, 2000], [2000, 2000, 2000, 2000], [2000, 2000, 1000, 1000], [1000, 1000, 2000, 2000],
            # [2000, 2000, 2000], [2000, 2000, 1000], [1000, 2000, 2000], [2000, 1000, 2000], [2000, 1000, 500], [500, 1000, 2000],
            # [3000, 2000], [2000, 3000], [2000, 1000], [1000, 2000], [2000, 500], [500, 2000],
            # [2000], [500]]
            suffixes = [pre_suffix + str(i) for i in range(suffix_start, suffix_start + len(layers))]
            for i in range(len(suffixes)):
                print('Model layers:', layers[i])
                if both:
                    train_both(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = layers[i],
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate,
                        show_plots = 0,
                        verbose = 0)
                else:
                    ModelInfo(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = layers[i],
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate).run(show_plots = 0, verbose = 0)
        elif 'training examples' == method:
            # Scan to establish how error scales with number of examples in the training set
            pre_suffix = '-f'
            training_set = [350, 600]# 500, 750, 1000, 1250, 1500, 1672, 1900]
            post_suffixes = [int(i/100) for i in training_set]
            suffixes = [pre_suffix + str(i) for i in post_suffixes]
            for i in range(len(suffixes)):
                print(training_set[i], 'examples in the training set.')
                if both:
                    train_both(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate,
                        training_examples = training_set[i],
                        show_plots = 0,
                        verbose = 0)
                else:
                    ModelInfo(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate,
                        training_examples = training_set[i]).run(show_plots = 0, verbose = 0)
        elif 'specialised' == method:
            pre_suffix = '-sp{}-l10'
            layers = []
            for i in range(12, 15, 2):
                layers += [[2000]*i]
            suffixes = [pre_suffix.format(str(i)) for i in range(suffix_start, suffix_start + len(layers))]
            for i in range(len(suffixes)):
                print('Model layers:', layers[i])
                if both:
                    raise NotImplementedError('')
                else:
                    model_info = ModelInfo(
                        datasets = datasets,
                        transfer_datasets = None,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = layers[i],
                        dropout = 0.2,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate,
                        resnet = False,
                        val_size = 0.20,
                        test_size = 10,
                        augmenting_data = True)
                    model_info.load_data(load_specialised=True, original_dataset=transfer_datasets)
                    model_info.run(show_plots = 0, verbose = 0, show_training_predictions = False, load = False)
        elif 'error' == method:
            # Scan to get a value for standard deviation
            pre_suffix = '-e'
            number_of_models = 20
            suffixes = [pre_suffix + str(i) for i in range(suffix_start, suffix_start + number_of_models)]
            for i in range(number_of_models):
                print('Model number', i)
                if both:
                    train_both(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate,
                        show_plots = 0,
                        verbose = 0)
                else:
                    ModelInfo(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = default_model_layers,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate).run(show_plots = 0, verbose = 0)
        elif 'test' == method:
            # Test value, usually to check for summary formatting
            pre_suffix = '-t'
            suffix = pre_suffix + str(suffix_start)
            print('Running test')
            epochs = 1
            default_model_layers = [2000]*7
            if both:
                train_both(
                    datasets = datasets,
                    transfer_datasets = transfer_datasets,
                    model_name = model_name,
                    suffix = suffix,
                    model_layers = default_model_layers,
                    epochs = epochs,
                    learning_rate = default_learning_rate,
                    l2_rate = default_l2_rate,
                    show_plots = 0,
                    verbose = 0)
            else:
                ModelInfo(
                    datasets = datasets,
                    transfer_datasets = transfer_datasets,
                    model_name = model_name,
                    suffix = suffix,
                    model_layers = default_model_layers,
                    dropout = 0,
                    epochs = epochs,
                    learning_rate = default_learning_rate,
                    l2_rate = 0,
                    resnet = False,
                    augmenting_data = False,
                    test_size = 0.01,
                    val_size = 0.01,
                    broadband_data = True,
                    training_examples = None).run(show_plots = 0, verbose = 1,
                                        show_training_predictions = False,
                                        load_sets = True)
        elif 'custom' == method:
            # Custom, often used to scan deep networks.
            # layers = [[2000]*14]
            # layers = [[2000]*12, [2000]*12, [2000]*12]
            # layers = [[2000]*7, ]
            layers = []
            for i in range(7, 20, 4):
                # layers += [[500]*i, [1000]*i, [2000]*i]
                layers += [[500]*i]
            pre_suffix = '-r-b'
            suffixes = [pre_suffix + str(i) for i in range(suffix_start, suffix_start + len(layers))]
            for i in range(len(suffixes)):
                print('Model layers:', layers[i])
                if both:
                    train_both(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = layers[i],
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = default_l2_rate,
                        show_plots = 0,
                        verbose = 0)
                else:
                    ModelInfo(
                        datasets = datasets,
                        transfer_datasets = transfer_datasets,
                        model_name = model_name,
                        suffix = suffixes[i],
                        model_layers = layers[i],
                        dropout = 0.2,
                        epochs = default_epochs,
                        learning_rate = default_learning_rate,
                        l2_rate = 0,
                        resnet = False,
                        augmenting_data = False,
                        test_size = 0.01,
                        val_size = 0.01,
                        broadband_data = True,
                        training_examples = 250000).run(show_plots = 0, verbose = 0,
                                            show_training_predictions = False,
                                            load_sets = True)
        else:
            print('Did not recognise paramater', method)
    return

def train_single():
    # Declare
    train_both(
        datasets = ('BigOne9', ),
        model_name = 'Themis',
        suffix = '',
        model_layers = [2000, 2000, 2000],
        epochs = 2000,
        learning_rate = 0.0001,
        l2_rate = 1*10**-6.8)

if socket.gethostname() == 'tauron':
    print('On Tauron so using GPU')
    set_gpu(0)

# plot_best_model_test_predictions()
assess_on_test_set()

# ModelInfo(
#     datasets = ('PracticalData9',),
#     model_name = 'Athena',
#     suffix = '-t3',
#     model_layers = [5000],
#     epochs = 2000,
#     learning_rate = 0.0001,
#     l2_rate = 1.5849e-7,
#     augmenting_data = False,
#     test_size = 0.02,
#     val_size = 0.02
# ).run(verbose=0, show_plots=0)

# ModelInfo(
#     datasets = ('PracticalData9',),
#     model_name = 'Athena',
#     suffix = '-t11',
#     model_layers = [5000],
#     epochs = 2000,
#     learning_rate = 0.0001,
#     l2_rate = 1.5849e-7,
#     augmenting_data = True,
#     test_size = 0.02,
#     val_size = 0.02
# ).run(verbose=0, show_plots=0)

# if __name__ == '__main__':
#     train(
#         datasets = ('PracticalData19_Spectrum',),
#         model_name = 'Gaia',
#         suffix_starts = [2,],
#         methods = ['custom',],
#         boths = [False,]
#     )

# if __name__ == '__main__':
#     train(
#         datasets = ('PracticalData19_Spectrum',),
#         model_name = 'Gaia',
#         suffix_starts = [1, ],
#         methods = ['test', ],
#         boths = [False, ]
#     )

# if __name__ == '__main__':
#     train(
#         datasets = ('PracticalData19_Spectrum',),
#         model_name = 'Gaia5W',
#         suffix_starts = [2,],
#         methods = ['custom',],
#         boths = [False,]
#     )

# if __name__ == '__main__':
#     train(
#         datasets = ('BigOne23',),
#         transfer_datasets = ('BigOne25', 'BigOne26', 'BigOne27', 'BigOne29', 'BigOne30'),
#         model_name = 'Hestia5',
#         suffix_starts = [9,],
#         methods = ['custom',],
#         boths = [False,]
#     )

# if __name__ == '__main__':
#     train(
#         datasets = ('BigOne31',),
#         transfer_datasets = ('BigOne32', 'BigOne34'),
#         model_name = 'Hestia10',
#         suffix_starts = [13,],
#         methods = ['custom',],
#         boths = [False,]
#     )

# if __name__ == '__main__':
#     train(
#         datasets = ('BigOne35',),
#         transfer_datasets = ('BigOne36',),
#         model_name = 'Hyperion5',
#         suffix_starts = [11,],
#         methods = ['custom',],
#         boths = [False,]
#     )

# if __name__ == '__main__':
#     train(
#         datasets = ('BigOne40',),
#         # transfer_datasets = ('BigOne31',),
#         model_name = 'Hestia10',
#         suffix_starts = [27,],
#         methods = ['custom',],
#         boths = [False,]
#     )
