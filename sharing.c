/*
============================================================================
Filename    : sharing.c
Author      : Madeline Desautel, Mathurin Verclytte
SCIPER		: 346981, 346153
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <omp.h>

int perform_bucket_computation(int, int, int);

int main (int argc, const char *argv[]) {
    int num_threads, num_samples, num_buckets;

    if (argc != 4) {
		printf("Invalid input! Usage: ./sharing <num_threads> <num_samples> <num_buckets> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        num_buckets = atoi(argv[3]);
	}
    
    set_clock();
    perform_buckets_computation(num_threads, num_samples, num_buckets);

    printf("Using %d threads: %d operations completed in %.4gs.\n", num_threads, num_samples, elapsed_time());
    return 0;
}

int perform_buckets_computation(int num_threads, int num_samples, int num_buckets) {

    typedef struct {
        int val;
        int padding[64 / sizeof(int) -1];
    } padded_counter;

    omp_set_num_threads(num_threads);
    rand_gen generator;
    volatile padded_counter histogram[num_buckets];
    for (int i = 0; i < num_buckets; ++i) {
        histogram->val = 0;
    }
    #pragma omp parallel default(none) shared(num_threads, num_samples, num_buckets, histogram) private(generator)
    {
        generator = init_rand();
        int threadArray[num_buckets];

        for (int i = 0; i < num_buckets; ++i) {
            threadArray[i] = 0;
        }

        #pragma omp for
        for (int i = 0; i < num_samples; i++) {
            int val = next_rand(generator) * num_buckets;
            threadArray[val]++;
        }
        free_rand(generator);
        for (int i = 0; i < num_buckets; ++i) {
            histogram[i].val += threadArray[i];
        }
    }
    return 0;
}