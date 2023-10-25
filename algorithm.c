/*
============================================================================
Filename    : algorithm.c
Author      : Madeline Desautel, Mathurin Verclytte
SCIPER      : 346981, 346153
============================================================================
*/
#include <math.h>
#include <omp.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

void simulate(double *input, double *output, int threads, int length, int iterations) {
    double *temp;
    double values[length*length/threads];
    int is[length*length/threads];
    int js[length*length/threads];
    int count;
    omp_set_num_threads(threads);

    for (int n = 0; n < iterations; n++) {

        #pragma omp parallel default(none) private(values, is, js, count) shared(iterations, input, output, length, temp, threads)
        {
            count = -1;
            for (int i = 0; i < length*length/threads; ++i) {
                is[i] = -1;
                js[i] = -1;
                values[i] = 0;
            }


            #pragma omp for collapse(2)
            for (int i = 1; i < length - 1; i++) {
                for (int j = 1; j < length - 1; j++) {
                    if (((i == length / 2 - 1) || (i == length / 2))
                        && ((j == length / 2 - 1) || (j == length / 2)))
                        continue;

                    count++;
                    is[count] = i;
                    js[count] = j;
                    values[count] = (INPUT(i - 1, j - 1) + INPUT(i - 1, j) + INPUT(i - 1, j + 1) +
                            INPUT(i, j - 1) + INPUT(i, j) + INPUT(i, j + 1) +
                            INPUT(i + 1, j - 1) + INPUT(i + 1, j) + INPUT(i + 1, j + 1)) / 9;
                }
            }
            
            for (int i = 0; i < length * length/threads; ++i) {

                if (is[i] == -1) break;

                #pragma omp atomic
                OUTPUT(is[i], js[i]) += values[i];
            }
        }
                temp = input;
                input = output;
                output = temp;
    }
}
