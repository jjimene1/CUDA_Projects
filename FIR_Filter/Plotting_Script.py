import numpy as np
import matplotlib.pyplot as plt

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = np.loadtxt(file)
    return data

def plot_magnitude(data1, data2, data3, data4):
    """
    Plot the magnitude of two sets of data.

    Parameters:
    - data1: NumPy array containing the first set of data.
    - data2: NumPy array containing the second set of data.
    """
    plt.figure(figsize=(10, 6))

    # Plot magnitude of data1
    plt.subplot(4, 1, 1)
    plt.plot(np.abs(data1), 'b-', label='Magnitude Data 1')
    plt.title('Unfiltered Data')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()

    # Plot magnitude of data2
    plt.subplot(4, 1, 2)
    plt.plot(np.abs(data2), 'r-', label='Magnitude Data 2')
    plt.title('Filtered Data')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()

    # Plot magnitude of data3
    plt.subplot(4, 1, 3)
    plt.plot(np.abs(data3), 'g-', label='Magnitude Data 2')
    plt.title('Filtered Data GPU')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()

    # Plot magnitude of data3
    plt.subplot(4, 1, 4)
    plt.plot(np.abs(data4), 'p-', label='Magnitude Data 3')
    plt.title('Filtered Data GPU')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define file names
    filename1 = "signal_data.txt"
    filename2 = "filtered_signal_CPU.txt"
    filename3 = "filtered_signal_GPU.txt"
    filename4 = "dft.txt"

    # Read data from files
    data1 = read_data_from_file(filename1)
    data2 = read_data_from_file(filename2)
    data3 = read_data_from_file(filename3)
    data4 = read_data_from_file(filename4)

    # Plot magnitude of the data
    plot_magnitude(data1, data2, data3, data4)