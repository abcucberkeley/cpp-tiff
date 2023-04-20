#include "parallelReadTiff.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

int
parse_console_args(int argc, char *argv[], char **file_name)
{
    int c, parse_code = -1;

    while ((c = getopt(argc, argv, "f:")) != -1) {
        switch (c) {
            case 'f':
                *file_name = optarg;
                parse_code = 0;
                break;
            default:
                fprintf(stderr, "Usage: %s [-f filename]\n", argv[0]);
                parse_code = -1;
                exit(EXIT_FAILURE);
        }
    }
    return parse_code;
}

int
main(int argc, char *argv[])
{

    char *file_name = NULL;
    void *tiff      = NULL;
    int   i         = 0;
    struct timespec start, end;
    double duration;

    // parse console argument
    int parse_code = parse_console_args(argc, argv, &file_name);
    if (parse_code) {
        return parse_code;
    }

    // print file name for validating purpose
    printf("Filename: %s\n", file_name ? file_name : "(none)");

    clock_gettime(CLOCK_MONOTONIC, &start); // start timing the operation

    // calling tiff loading process.
    parallel_TIFF_load(file_name, &tiff, 1, NULL);

    clock_gettime(CLOCK_MONOTONIC, &end); // end timing the operation

    duration = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec); // calculate duration in nanoseconds

    printf("Read Tiff File Done! Time taken: %.4f seconds\n", duration/1e9);

    if (!tiff)
        return 1;

    printf("first few bytes ");
    for (i = 0; i < 10; i++) {
        printf("%d ", ((uint8_t *)tiff)[i]);
    }
    free(tiff);
    return 0;
}
