#ifndef PARALLELREADTIFF_H
#define PARALLELREADTIFF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>

typedef struct {
    uint64_t *range;
    size_t    length;
} parallel_tiff_range_t;

/**
 * This function loads a TIFF image file in parallel into memory, and returns a pointer to the loaded data.
 * The image data is stored in the memory pointed to by tiff_ptr, and the dimensions of the image are stored
 * in strip_range. If flipXY is set to a non-zero value, the image dimensions will be swapped. The function
 * also takes in a file name parameter to specify the location of the TIFF image file to load.
 * 
 * @param fileName: A pointer to a string that specifies the location of the TIFF image file to load.
 * @param tiff_ptr: A pointer to a pointer that will be used to return the memory location of the loaded image data.
 * @param flipXY: An unsigned 8-bit integer that specifies whether to swap the X and Y dimensions of the image.
 * @param strip_range: A pointer to a parallel_tiff_range_t struct that will be used to return the dimensions of the image.
 * 
 * @return This function does not return a value, but the loaded image data and image dimensions are returned via the
 *         tiff_ptr and strip_range parameters, respectively.
 */
void parallel_TIFF_load(char *fileName, void **tiff_ptr, uint8_t flipXY, parallel_tiff_range_t *strip_range);

#endif // PARALLELREADTIFF_H