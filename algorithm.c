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
    omp_set_num_threads(threads);

    for (int n = 0; n < iterations; n++) {

#pragma omp parallel default(none) shared(iterations, input, output, length, temp, threads)
        {

#pragma omp for
            for (int i = 1; i < length - 1; i++) {
                for (int j = 1; j < length - 1; j += 2) {

                    if (((i == length / 2 - 1) || (i == length / 2))
                        && ((j == length / 2 - 1) || (j == length / 2)))
                        continue;

                    double c1 = INPUT(i - 1, j - 1);
                    double c2 = INPUT(i - 1, j);
                    double c3 = INPUT(i - 1, j + 1);
                    double c4 = INPUT(i, j - 1);
                    double c5 = INPUT(i, j);
                    double c6 = INPUT(i, j + 1);
                    double c7 = INPUT(i + 1, j - 1);
                    double c8 = INPUT(i + 1, j);
                    double c9 = INPUT(i + 1, j + 1);


                    OUTPUT(i,j) = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9) / 9;

                    if (((i == length / 2 - 1) || (i == length / 2))
                        && ((j +1 == length / 2 - 1) || (j +1 == length / 2)))
                        continue;

                    double c10 = INPUT(i - 1, j + 2);
                    double c11 = INPUT(i, j + 2);
                    double c12 = INPUT(i + 1, j + 2);

                    OUTPUT(i,j+1) = (c2 + c3 + c10 +
                                     c5 + c6 + c11 +
                                     c8 + c9 + c12) / 9;

                }
            }

        }
        temp = input;
        input = output;
        output = temp;
    }
}
