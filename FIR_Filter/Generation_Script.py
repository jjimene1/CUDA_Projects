import numpy as np

def generate_synthetic_data(num_samples, num_taps):
    # Generate random signal data
    signal = np.random.rand(num_samples).astype(np.float32)

    # Generate random filter coefficients
    filter_coeffs = np.random.rand(num_taps).astype(np.float32)

    return signal, filter_coeffs

def generate_sine_wave(num_samples, frequency, amplitude=1.0, phase=0.0):
    # Generate a sine wave signal
    t = np.arange(num_samples)
    signal = amplitude * np.sin(2 * np.pi * frequency * t / num_samples + phase)
    return signal.astype(np.float32)

def write_signal_to_file(filename, filter_coeffs):
    np.savetxt(filename, filter_coeffs)

if __name__ == "__main__":
    # Define parameters
    num_samples = 8192
    num_taps = 2048
    signal_output_file = "signal_data.txt"
    filter_coeffs_output_file = "filter_coeffs.txt"
    sine_wave_file = "sine_data.txt"

    # Generate synthetic data
    signal, filter_coeffs = generate_synthetic_data(num_samples, num_taps)
    sine_signal = generate_sine_wave(num_samples, 50)

    # Write generated data to files
    write_signal_to_file(signal_output_file, signal)
    write_signal_to_file(filter_coeffs_output_file, filter_coeffs)
    write_signal_to_file(sine_wave_file, sine_signal)

    print(f"Synthetic signal data has been written to '{signal_output_file}'.")
    print(f"Synthetic filter coefficients have been written to '{filter_coeffs_output_file}'.")