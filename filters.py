# Thomas Roberts
# CS 4732 Section 01: Machine Vision
# Professor Karakaya
# February 23, 2024
#
# last modified February 25, 2024

import numpy

class Morphological_Spatial_Filters:
    """
        Description:
            This class contain functions to enhance image arrays.

        Functions
        ----------
        laplace_filter
            function to sharpen an image using the laplace filter
        sobel_filter
            A function to sharpen an image using the sobel operation
        hit
            A support function to determine if a structuring element hits a selection of pixels
        fit
            A support function to determine if a structuring elements fits a selection of pixels

    """
    STRUCT_ELEMENT = [numpy.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                    numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                    numpy.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], 
                                   [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])]

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

            Note: I found that the original sobel filter alone was a very sharp, so I added the
            coefficient variable so user's can control the degree of sharpness.

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

    def dilution(orig_image: numpy.ndarray, struct_ID: int) -> numpy.ndarray:
        """
            Function to dilate a binary image given a structure element id

            Paramaters
            ----------
            orig_image
                original image array to be eroded
            struct_ID
                The number ID of a structuring element constant

            Returns
            --------
            NDArray
                eroded image array
        """
        # dimension of image
        max_row, max_col = orig_image.shape

        # getting halfwidth
        struct = Morphological_Spatial_Filters.STRUCT_ELEMENT[struct_ID]
        halfwidth = len(struct)//2

        # creating space for new enhanced image
        dilated_copy = numpy.zeros(orig_image.shape, dtype=bool)

        for row in range(halfwidth, max_row-halfwidth):
            for col in range(halfwidth, max_col-halfwidth):
                # creating subarray of orig_image to check if fits structuring element
                region_selection = numpy.zeros(struct.shape)
                for index in range(len(struct)):
                    region_selection[index] = orig_image[row + (index-halfwidth)][col-halfwidth:col+halfwidth+1]

                # calling fit function 
                pixel_value = Morphological_Spatial_Filters.hit(region_selection, struct_ID)
                dilated_copy[row][col] = pixel_value
        
        return dilated_copy
    # End of function erosion

    def erosion(orig_image: numpy.ndarray, struct_ID: int) -> numpy.ndarray:
        """
            Function to erode a binary image given a structure element id

            Paramaters
            ----------
            orig_image
                original image array to be eroded
            struct_ID
                The number ID of a structuring element constant

            Returns
            --------
            NDArray
                eroded image array
        """
        # dimension of image
        max_row, max_col = orig_image.shape

        # getting halfwidth
        struct = Morphological_Spatial_Filters.STRUCT_ELEMENT[struct_ID]
        halfwidth = len(struct)//2

        # creating space for new enhanced image
        eroded_copy = numpy.zeros(orig_image.shape, dtype=bool)

        for row in range(halfwidth, max_row-halfwidth):
            for col in range(halfwidth, max_col-halfwidth):
                # creating subarray of orig_image to check if fits structuring element
                region_selection = numpy.zeros(struct.shape)
                for index in range(len(struct)):
                    region_selection[index] = orig_image[row + (index-halfwidth)][col-halfwidth:col+halfwidth+1]

                # calling fit function 
                pixel_value = Morphological_Spatial_Filters.fit(region_selection, struct_ID)
                eroded_copy[row][col] = pixel_value
        
        return eroded_copy
    # End of function erosion

    def binary(orig_image: numpy.ndarray, threshold: int = 185) -> numpy.ndarray:
        """
            Converts a grayscale image to binary image based on the threshold provided.

            Paramaters
            -----------
            orig_image: NDarray
                array of original image
            threshold: int
                threshold value 

            Returns
            -------
            NDarray
                new binary image
        """
        # getting dimensions
        max_row, max_column = orig_image.shape[:2]

        # if image is color, converting it grayscale
        if len(orig_image.shape) > 2:
        orig_image = Morphological_Spatial_Filters.grayscale(orig_image)

        # creating space for new array
        binary_image = numpy.zeros(orig_image.shape, dtype = bool)

        # applying binary threshold
        for row in range(max_row):
            for col in range(max_column):
                if orig_image[row][col] > threshold:
                    binary_image[row][col] = 1
                else:
                    binary_image[row][col] = 0

        return binary_image
    # End of function binary

    def grayscale(orig_image: numpy.ndarray) -> numpy.ndarray:
        """
            A function to convert a color image to grayscale

            Paramaters
            ----------
            orig_image: NDarray
                image array of original image

            Returns
            --------
            NDarray
                grayscale image array
        """
        # getting dimensions
        max_row, max_col = orig_image.shape[:2]

        # creating space for copy
        gray_image = numpy.zeros((max_row, max_col), dtype = "uint8")

        # avering color pixels into single gray pixel
        for row in range(max_row):
            for col in range(max_col):
                gray_image[row][col] = numpy.average(orig_image[row][col])

        return gray_image
    # End of function grayscale

    def hit(pixel_region: numpy.ndarray, struct_ID: int) -> bool:
        """
            Returns whether a structuring element will hit or miss in a region

            Paramaters
            -----------
            pixel_region: NDArray
                A square matrix of pixels to compare to a structuring element constant
            struct_ID
                The number id associated with structuring element constants listed

            Returns
            -------
            bool
                True if any pixels in the struct element hit in the region, false otherwise
        """
        gridWidth = len(pixel_region)

        struct = Morphological_Spatial_Filters.STRUCT_ELEMENT[struct_ID]

        # if shapes don't match, it will create index-out-bounds later
        if not struct.shape == pixel_region.shape:
            error_content = "Not Comparable. Pixel_region.shape must match struct.shape"
            error_content += "\n\nstructure shape: " + str(struct.shape)
            error_content += "\npixel_region shape: " + str(pixel_region.shape)
            raise Exception(error_content)
        
        # only 1 pixel needs to hit for function to return True
        for row in range(gridWidth):
            for col in range(gridWidth):
                if struct[row][col] == pixel_region[row][col] == 1:
                    return True
                
        return False
    # End of function hit

    def fit(pixel_region: numpy.ndarray, struct_ID: int) -> bool:
        """
            Returns whether a structuring element will fit or miss in a region

            Paramaters
            -----------
            pixel_region: NDArray
                A square matrix of pixels to compare to a structuring element constant
            struct_ID
                The number id associated with structuring element constants listed

            Returns
            -------
            bool
                True if all pixels in the struct element hit in the region, false otherwise
        """
        gridWidth = len(pixel_region)

        struct = Morphological_Spatial_Filters.STRUCT_ELEMENT[struct_ID]

        # if shapes don't match, it will create index-out-bounds later
        if not struct.shape == pixel_region.shape:
            error_content = "Not Comparable. Pixel_region.shape must match struct.shape"
            error_content += "\n\nstructure shape: " + str(struct.shape)
            error_content += "\npixel_region shape: " + str(pixel_region.shape)
            raise Exception(error_content)
        
        # all pixels needs to hit for function to return True
        for row in range(gridWidth):
            for col in range(gridWidth):
                if struct[row][col] == 1:
                    if not pixel_region[row][col] == 1:
                        return False
                
        return True
    # End of function hit

    def min_nonzero(matrix: numpy.ndarray) -> int:
        """
            Returns the lowest nonzero number from a square matrix
        """
        max_row, max_col = matrix.shape[:2]
        
        lowest = numpy.max(matrix)

        for row in range(max_row):
            for col in range(max_col):
                if not matrix[row][col] == 0:
                    if matrix[row][col] < lowest:
                        lowest = matrix[row][col]
        return lowest
# End of class Morphological_Spatial_Filters