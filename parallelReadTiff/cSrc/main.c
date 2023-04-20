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
    // parse console argument
    int parse_code = parse_console_args(argc, argv, &file_name);
    if (parse_code) {
        return parse_code;
    }

    // print file name for validating purpose
    printf("Filename: %s\n", file_name ? file_name : "(none)");

    clock_t begin = clock();
    // calling tiff loading process.
    parallel_TIFF_load(file_name, &tiff, 1, NULL);

    if (!tiff)
        return 1;
    clock_t end        = clock();
    double  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Read Tiff File Done! Time spent: %.4f seconds\n", time_spent);

    printf("first few bytes ");
    for (i = 0; i < 10; i++) {
        printf("%d ", ((uint8_t *)tiff)[i]);
    }
    free(tiff);
    return 0;
}
