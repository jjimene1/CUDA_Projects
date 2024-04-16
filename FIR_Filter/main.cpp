#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>  
#include <complex.h>
#include <math.h>

#define MAX_SAMPLES 8192
#define MAX_TAPS 1024

struct FIRFilter{
    float buf[MAX_TAPS];
    uint16_t bufIndex;
    float out;
};

void FIRFilter_Init(FIRFilter *fir){
    for (uint16_t i = 0; i < MAX_TAPS; i++){
        fir->buf[i] = 0.0f;
    }

    fir->out = 0.0f;
}

float FIRFilter_Update(FIRFilter *fir, float inp, float *filter_coeffs){
    // Store the latest sample in the buffer
    fir->buf[fir->bufIndex] = inp;
    //Increment buffer index and wrap around if necessary (circular buffer)
    fir->bufIndex++;

    if (fir->bufIndex == MAX_TAPS){
        fir->bufIndex = 0;
    }

    //Computing new output sample (convolution)
    fir->out = 0.0f;

    uint16_t sumIndex = fir->bufIndex;

    //decrement index and wrap if necessary 
    for(uint16_t i = 0; i < MAX_TAPS; i++){
        if (sumIndex > 0){
            sumIndex--;
        }
        else {
            sumIndex = MAX_TAPS - 1;
        }

        fir->out += filter_coeffs[i] * fir->buf[sumIndex];
    }


    return fir->out;
} 

//DFT Experiments
//naive DFT Implementation
void dft(float _Complex* Input, float _Complex* Output, int samples){
    for (int k = 0; k < samples; k++){
        float _Complex sum = 0;

        //value of exponent for that row
        float c = -2 * M_PI * k / samples;
        for(int n = 0; n < samples; n++){
            //Value of exponent for that column
            float w = c * n;

            //Euler's formula
            // e^ix = cos x + i sin x

            // Compute input[n] * exp(-2i pi * k * n / N)
            sum = sum + Input[n] * (ccos(w) + I * csin(w));
        }
        Output[k] = sum;
    }
}

//helper function to reverse bits for FFT
uint32_t reverse_bits(uint32_t x)
{
    // 1. Swap the position of consecutive bits
    // 2. Swap the position of consecutive pairs of bits
    // 3. Swap the position of consecutive quads of bits
    // 4. Continue this until swapping the two consecutive 16-bit parts of x
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

//FFT
void fft(const float _Complex* input, float _Complex* output, uint32_t samples)
{
    int logN = (int) log2f((float) samples);

    for (uint32_t i = 0; i < samples; i++) {
        // Reverse the 32-bit index.
        uint32_t rev = reverse_bits(i);

        // Only keep the last logN bits of the output.
        rev = rev >> (32 - logN);

        // Base case: set the output to the bit-reversed input.
        output[i] = input[rev];
    }

    // Set m to 2, 4, 8, 16, ..., N
    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int mh = 1 << (s - 1);

        float _Complex twiddle = cexpf(-2.0I * M_PI / m);

        // Iterate through Y in strides of length m=2**s
        // Set k to 0, m, 2m, 3m, ..., N-m
        for (uint32_t k = 0; k < samples; k += m) {
            float _Complex twiddle_factor = 1;

            // Set both halves of the Y array at the same time
            // j = 1, 4, 8, 16, ..., N / 2
            for (int j = 0; j < mh; j++) {
                float _Complex a = output[k + j];
                float _Complex b = twiddle_factor * output[k + j + mh];

                // Compute pow(twiddle, j)
                twiddle_factor *= twiddle;

                output[k + j] = a + b;
                output[k + j + mh] = a - b;
            }
        }

    }
}


//helper complex transformation function
float _Complex* float_to_complex(float* real_array, int length) {
    float _Complex* complex_array = (float _Complex*)malloc(length * sizeof(float _Complex));
    if (complex_array == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    for (int i = 0; i < length; i++) {
        complex_array[i] = real_array[i]; // Assign real part
    }

    return complex_array;
}


void read_signal_from_file(const char *filename, float *signal, int num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read signal data
    for (int i = 0; i < num_samples; i++) {
        fscanf(file, "%e", &signal[i]);
    }

    fclose(file);
}

void write_signal_to_file(const char *filename, float *signal, int num_samples) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write signal data
    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "%f\n", signal[i]);
    }

    fclose(file);
}

int main() {
    // Define arrays to store data
    float signal[MAX_SAMPLES];
    float filter_coeffs[MAX_TAPS];
    float filtered_signal[MAX_SAMPLES];
    FIRFilter filter_data;

    // Define file names
    const char *signal_filename = "signal_data.txt";
    const char *filter_coeffs_filename = "filter_coeffs.txt";
    const char *filtered_signal_filename = "filtered_signal.txt";
    const char *sine_signal = "sine_data.txt";
    const char *dft_filename = "dft.txt";

    // Read data from files
    read_signal_from_file(signal_filename, signal, MAX_SAMPLES);
    read_signal_from_file(filter_coeffs_filename, filter_coeffs, MAX_TAPS);

    //Start the FIR Filter Struct
    FIRFilter_Init(&filter_data);

    //Declare variables for CPU timing
    clock_t start_cpu, end_cpu;
    double cpu_time_used;
    printf("Carrying out filtering in CPU...\n");
    start_cpu = clock();

    //Carry out filtering for the synthethic data
    for(int i = 0; i < MAX_SAMPLES; i++){
        FIRFilter_Update(&filter_data, signal[i], filter_coeffs);
        filtered_signal[i] = filter_data.out;
    }
    end_cpu = clock();

    cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000; //measuring in ms

    // Print CPU execution time
    printf("Computation time in CPU = %.3f ms\n", cpu_time_used);
    //double cpu_gflops = numOps / (cpu_time_used * 1e6);
    //printf("Performance on CPU = %.2f GFlops/s \n", cpu_gflops);

    //Do DFT 
    float sine_data[MAX_SAMPLES];
    read_signal_from_file(sine_signal, sine_data, MAX_SAMPLES);
    float _Complex* signal_complex = float_to_complex(sine_data, MAX_SAMPLES);
    float _Complex DFT_output[MAX_SAMPLES];
    clock_t start_dft = clock();
    //dft(signal_complex, DFT_output, MAX_SAMPLES);
    uint32_t samples = MAX_SAMPLES;
    fft(signal_complex, DFT_output, samples);
    clock_t end_dft = clock();

    double dft_time_used = ((double) (end_dft - start_dft)) / CLOCKS_PER_SEC * 1000;
    printf("Computation time for DFT = %.3f ms\n", dft_time_used);
    float DFT_magnitude[MAX_SAMPLES];
    for(int i = 0; i < MAX_SAMPLES; i++){
        DFT_magnitude[i] = cabs(DFT_output[i]);
    }

    write_signal_to_file(filtered_signal_filename, filtered_signal, MAX_SAMPLES);
    write_signal_to_file(dft_filename, DFT_magnitude, MAX_SAMPLES);

    free(signal_complex);
    return 0;
}