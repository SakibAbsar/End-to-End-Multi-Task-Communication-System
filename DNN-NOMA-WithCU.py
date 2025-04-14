# DNN-NOMA-WithCU
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load MNIST dataset
(x_train_CU, y_train_CU), (x_test_CU, y_test_CU) = mnist.load_data()

# Flatten the data and normalize
x_train_CU = x_train_CU.reshape((60000, 28 * 28)).astype('float32') / 255.0
x_test_CU = x_test_CU.reshape((10000, 28 * 28)).astype('float32') / 255.0

# Define the AWGN function
def add_awgn_noise_CU(signal_CU, snr_db_CU):
    """Add AWGN noise to the signal with a given SNR in dB."""
    snr_CU = 10 ** (snr_db_CU / 10.0)
    signal_power_CU = tf.reduce_mean(tf.square(signal_CU))
    noise_power_CU = signal_power_CU / snr_CU
    noise_CU = tf.random.normal(shape=tf.shape(signal_CU), mean=0.0, stddev=tf.sqrt(noise_power_CU))
    return signal_CU + noise_CU

# Binary classification labels for Task 1 (digit "2")
y_train_binary_CU = (y_train_CU == 2).astype('float32')
y_test_binary_CU = (y_test_CU == 2).astype('float32')

# Categorical classification labels for Task 2
y_train_categorical_CU = to_categorical(y_train_CU, 10)
y_test_categorical_CU = to_categorical(y_test_CU, 10)

# Create sample weights for the binary classification task
class_weights_CU = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_binary_CU), y=y_train_binary_CU)
sample_weights_binary_CU = np.where(y_train_binary_CU == 1, class_weights_CU[1], class_weights_CU[0])

# Define the CU Encoder
class CUEncoder_CU(tf.keras.Model):
    def __init__(self):
        super(CUEncoder_CU, self).__init__()
        self.flatten_CU = layers.Flatten()
        self.fc_CU = layers.Dense(64, activation='tanh')
    
    def call(self, inputs_CU):
        x_CU = self.flatten_CU(inputs_CU)
        x_CU = self.fc_CU(x_CU)
        return x_CU

# SU1 Encoder for Task 1
class SU1_CU(tf.keras.Model):
    def __init__(self):
        super(SU1_CU, self).__init__()
        self.encoder_CU = layers.Dense(16, activation='tanh')
    
    def call(self, inputs_CU):
        x_CU = self.encoder_CU(inputs_CU)
        return x_CU

# SU2 Encoder for Task 2
class SU2_CU(tf.keras.Model):
    def __init__(self):
        super(SU2_CU, self).__init__()
        self.encoder_CU = layers.Dense(16, activation='tanh')
    
    def call(self, inputs_CU):
        x_CU = self.encoder_CU(inputs_CU)
        return x_CU

# Digital Modulation (BPSK)
def bpsk_modulate_CU(signal_CU):
    return 2 * signal_CU - 1  # Maps 0 to -1 and 1 to +1

# Differentiable BPSK Demodulation
def bpsk_demodulate_CU(received_signal_CU):
    """
    Perform differentiable BPSK demodulation.
    """
    return tf.sigmoid(received_signal_CU * 10)  # Approximates step function for soft decisions

# Superimpose signals using NOMA with different power levels
def noma_superimpose_CU(x1_CU, x2_CU, power1_CU=0.6, power2_CU=0.4):
    """Superimpose two signals using different power levels as per NOMA."""
    return tf.sqrt(power1_CU) * x1_CU + tf.sqrt(power2_CU) * x2_CU

# Define the DNN-based base station for signal separation
class BaseStation_CU(tf.keras.Model):
    def __init__(self):
        super(BaseStation_CU, self).__init__()
        self.dnn_CU = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu')
        ])
        self.su1_decoder_CU = layers.Dense(1, activation='sigmoid', name='binary_output_CU')
        self.su2_decoder_CU = layers.Dense(10, activation='softmax', name='categorical_output_CU')
    
    def call(self, inputs_CU):
        x_CU = self.dnn_CU(inputs_CU)
        su1_output_CU = self.su1_decoder_CU(x_CU)  # Binary classification output
        su2_output_CU = self.su2_decoder_CU(x_CU)  # Categorical classification output
        return {'binary_output_CU': su1_output_CU, 'categorical_output_CU': su2_output_CU}

# Define the overall Multi-Task NOMA-based communication model
class MultiTaskNOMA_CU(tf.keras.Model):
    def __init__(self, snr_db_CU):
        super(MultiTaskNOMA_CU, self).__init__()
        self.cu_encoder_CU = CUEncoder_CU()
        self.su1_CU = SU1_CU()
        self.su2_CU = SU2_CU()
        self.base_station_CU = BaseStation_CU()
        self.snr_db_CU = snr_db_CU
    
    def call(self, inputs_CU):
        # CU Encoder
        cu_output_CU = self.cu_encoder_CU(inputs_CU)
        
        # Process through SU1 and SU2 Encoders
        su1_output_CU = self.su1_CU(cu_output_CU)
        su2_output_CU = self.su2_CU(cu_output_CU)
        
        # BPSK Modulation
        su1_modulated_CU = bpsk_modulate_CU(su1_output_CU)
        su2_modulated_CU = bpsk_modulate_CU(su2_output_CU)
        
        # NOMA Superimpose
        superimposed_signal_CU = noma_superimpose_CU(su1_modulated_CU, su2_modulated_CU)
        
        # Pass through AWGN channel
        noisy_signal_CU = add_awgn_noise_CU(superimposed_signal_CU, self.snr_db_CU)
        
        # Demodulation Step: Differentiable BPSK Demodulation
        demodulated_signal_CU = bpsk_demodulate_CU(noisy_signal_CU)
        
        # Base Station separates and decodes the signals
        return self.base_station_CU(demodulated_signal_CU)

# List of SNR values
snr_values_CU = [-10, -5, 0, 5, 10, 15, 20]
binary_task_error_rates_CU = []
categorical_task_error_rates_CU = []


# Iterate over different SNR values
for snr_db_CU in snr_values_CU:
    print(f"Running the model for SNR: {snr_db_CU} dB")
    
    # Instantiate the model with the current SNR
    model_CU = MultiTaskNOMA_CU(snr_db_CU)
    
    # Compile the model
    model_CU.compile(optimizer='adam',
                  loss={'binary_output_CU': 'binary_crossentropy', 'categorical_output_CU': 'categorical_crossentropy'},
                  metrics={'binary_output_CU': 'accuracy', 'categorical_output_CU': 'accuracy'})

    # Train the model for the current SNR value
    history_CU = model_CU.fit(x_train_CU, {'binary_output_CU': y_train_binary_CU, 'categorical_output_CU': y_train_categorical_CU},
                        sample_weight={'binary_output_CU': sample_weights_binary_CU, 'categorical_output_CU': np.ones_like(y_train_binary_CU)},
                        epochs=30,
                        batch_size=128,
                        validation_data=(x_test_CU, {'binary_output_CU': y_test_binary_CU, 'categorical_output_CU': y_test_categorical_CU}),
                        verbose=0)

    # Evaluate accuracy for both tasks
    evaluation_results_CU = model_CU.evaluate(x_test_CU, {'binary_output_CU': y_test_binary_CU, 'categorical_output_CU': y_test_categorical_CU}, verbose=0)
    binary_accuracy_CU = evaluation_results_CU[1]
    categorical_accuracy_CU = evaluation_results_CU[2]
    
    # Calculate Task Error Rate
    binary_task_error_rate_CU = 1 - binary_accuracy_CU
    categorical_task_error_rate_CU = 1 - categorical_accuracy_CU
    binary_task_error_rates_CU.append(binary_task_error_rate_CU)
    categorical_task_error_rates_CU.append(categorical_task_error_rate_CU)

    print(f"SNR: {snr_db_CU} dB - Binary Task Error Rate: {binary_task_error_rate_CU:.5f}")
    print(f"SNR: {snr_db_CU} dB - Categorical Task Error Rate: {categorical_task_error_rate_CU:.5f}")

# Plot Task Error Rate vs SNR
plt.figure(figsize=(8, 6))
plt.plot(snr_values_CU, binary_task_error_rates_CU, marker='o', color='r', label='Binary Task Error Rate')
plt.plot(snr_values_CU, categorical_task_error_rates_CU, marker='x', color='g', label='Categorical Task Error Rate')
plt.title('Task Error Rate vs SNR for Binary and Categorical Tasks')
plt.xlabel('SNR (dB)')
plt.ylabel('Task Error Rate')
plt.grid(True)
plt.legend()
plt.show()

# Save Task Error Rate Data
task_error_rate_data = pd.DataFrame({
    'SNR (dB)': snr_values_CU,
    'Binary Task Error Rate': binary_task_error_rates_CU,
    'Categorical Task Error Rate': categorical_task_error_rates_CU
})
task_error_rate_data.to_csv('NOMA_WithCU_task_error_rate_data.csv', index=False)

print("Graph data saved to 'NOMA_WithCU_task_error_rate_data.csv'.")
