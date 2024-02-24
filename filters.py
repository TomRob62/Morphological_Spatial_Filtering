# Thomas Roberts
# CS 4732 Section 01: Machine Vision
# Professor Karakaya
# February 23, 2024
#
# last modified February 23, 2024

import numpy

class Morphological_Spatial_Filters:
    """
        Description:
            This class contain functions to enhance image arrays.

        Functions
        ----------
        laplace_filter
            function to sharpen an image using the laplace filter

    """

    def laplace_filter(orig_image: numpy.ndarray)->numpy.ndarray:
        """
            A function to sharpen an image using the formula described below:

                f(x, y) = 5*f(x, y) - f(x+1, y) - f(x-1, y) - f(x, y+1) - f(x, y-1) 

            Paramaters
            ----------
            orig_image: NDarray
                image array of original image

            Returns
            --------
            NDarray
                enhanced image array
        """
        # dimension of image
        max_row, max_col = orig_image.shape

        # creating space for new enhanced image
        enhanced_copy = numpy.zeros(orig_image.shape, dtype="uint8")

        # changing orig_image dtype because overflow error when applying formula
        orig_image = orig_image.astype("uint16")

        for row in range(1, max_row-1):
            for col in range(1, max_col-1):
                # applying formula 
                pixel_value = 5*(orig_image[row][col]) - orig_image[row+1][col] - orig_image[row-1][col] - \
                orig_image[row][col+1] - orig_image[row][col-1]

                # restricting maximum pixel value to prevent overflow when assigning value
                if pixel_value > 255:
                    pixel_value = 255

                # applying absolute value to prevent unsisgned overflow
                enhanced_copy[row][col] = abs(pixel_value)
        
        return enhanced_copy
    # End of definition laplace_filter

    def sobel_filter(orig_image: numpy.ndarray, coefficient: float = .3) -> numpy.ndarray:
        """
            A function to sharpen an image using the operations described below:

                step 1: calculate the horizontal and vertical operations of the sobel:
                    sobel_horizontal = f(x-1,y+1) + 2*f(x,y+1) + f(x+1,y+1) -
                                    (f(x-1, y-1) + 2*f(x, y-1) + f(x+1, y-1))
                    sobel_vertical = f(x+1, y-1) + 2*f(x+1, y) +f(x+1, y+1) -
                                    (f(x-1, y-1) + f(x-1, y) + f(x-1, y+1))

                step 2: subtract sobel picture from original image

            Paramaters
            ----------
            orig_image: NDarray
                image array of original image
            coefficient: float (optional)
                degree of contrast. default value is 0.3

            Returns
            --------
            NDarray
                enhanced image array
        """
        # dimension of image
        max_row, max_col = orig_image.shape

        # creating space for new enhanced image
        enhanced_copy = numpy.zeros(orig_image.shape, dtype="uint8")

        # changing orig_image dtype because overflow error when applying formula
        orig_image = orig_image.astype("uint16")

        for row in range(1, max_row-1):
            for col in range(1, max_col-1):
                # applying step 1 formulas
                sobel_horizontal_bottom = orig_image[row+1][col-1] + 2*orig_image[row+1][col] + orig_image[row+1][col+1] - \
                    (orig_image[row-1][col-1] + 2*orig_image[row-1][col] + orig_image[row-1][col+1])
                sobel_vertical_left = orig_image[row-1][col-1] + 2*orig_image[row][col-1] + orig_image[row+1][col-1] - \
                    (orig_image[row-1][col+1] + 2*orig_image[row][col+1] + orig_image[row+1][col+1])
                
                # step 2
                pixel_value = orig_image[row][col] - coefficient*(sobel_vertical_left + sobel_horizontal_bottom)

                # restricting maximum pixel value to prevent overflow when assigning value
                if pixel_value > 255:
                    pixel_value = 255
                if pixel_value < 0:
                    pixel_value = 0

                enhanced_copy[row][col] = pixel_value
        
        return enhanced_copy
    # End of function sobel_filter
# End of class Morphological_Spatial_Filters