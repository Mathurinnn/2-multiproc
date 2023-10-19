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
#define PADDED(I,J) padded[(I)*length+(J)]
#define INPUTPADDED(I,J) inputpadded[(I)*length+(J)].val

typedef struct {
    double val;
    double padding[64 / sizeof(double) -1];
} padded_struct;

void simulate(double *input, double *output, int threads, int length, int iterations) {
    double *temp;
    double padded[length*length];
    padded_struct inputpadded[length*length];
    int is[length*length];
    int js[length*length];
    int count;
    omp_set_num_threads(threads);

    for (int n = 0; n < iterations; n++) {

        #pragma omp parallel default(none) private(padded, is, js, count) shared(iterations, input, output, length, temp, inputpadded)
        {
            count = -1;
            for (int i = 0; i < length; ++i) {
                for (int j = 0; j < length; ++j) {
                    PADDED(i,j) = 0;
                    is[i*length+j] = -1;
                    js[i*length+j] = -1;
                }
            }

#pragma omp single
            {
                for (int i = 0; i < length; ++i) {
                    for (int j = 0; j < length; ++j) {
                        INPUTPADDED(i, j) = INPUT(i, j);
                    }
                }
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
                    PADDED(i,j) = (INPUTPADDED(i - 1, j - 1) + INPUTPADDED(i - 1, j) + INPUTPADDED(i - 1, j + 1) +
                            INPUTPADDED(i, j - 1) + INPUTPADDED(i, j) + INPUTPADDED(i, j + 1) +
                            INPUTPADDED(i + 1, j - 1) + INPUTPADDED(i + 1, j) + INPUTPADDED(i + 1, j + 1)) / 9;
                }
            }
            
            for (int i = 0; i < length * length; ++i) {

                if (is[i] == -1) break;

                #pragma omp atomic
                OUTPUT(is[i], js[i]) += PADDED(is[i], js[i]);
            }
        }
                temp = input;
                input = output;
                output = temp;
    }
}
