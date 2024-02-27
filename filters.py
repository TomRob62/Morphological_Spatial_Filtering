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

    def connected_comp(orig_image: numpy.ndarray, struct_ID:int  = 1) -> list:
        """
            A function that finds connected components and returns:
             1. an image where each component has a different gray value
             2. a matrix of orig_image.shape. Contains int ID of each pixel
             3. table of associative values
             4. frequence of each row in the table

                Uses two pass algorithm to ID each pixel in a single component has 
                the same ID, and component has different ID.

            Paramaters
            -----------
            orig_image: NDArray
                the original image array
            struct_ID: int
                The class id of the structure used to define the connectivity

            Returns
            -------
            conponents_image: NDArray
                A new image where each component has a unique grayscale value.
            conn_comp: NDarray
                a matrix of every ID corresponding to original image.
            table: 2D list
                a matrix where each row contains equivalent labels. each row = 1 component.
            frequency: list
                a list representing the number of pixels per component.
        """
        # getting dimensions
        max_row, max_column = orig_image.shape[:2]
        struct_width = len(Morphological_Spatial_Filters.STRUCT_ELEMENT[struct_ID])
        half_width = struct_width//2

        # if image is not binary, converting it to binary
        if not orig_image.dtype == bool:
            orig_image = Morphological_Spatial_Filters.binary(orig_image)

        # creating space for conn_comp array. holds pixel ID's
        conn_comp = numpy.zeros((max_row, max_column), dtype="uint32")

        # creating table to hold associative (equivalent) labels
        table = []

        # id variable for labeling pixels with no id or id'd neighbors
        pixel_id = 1

        # first pass
        for row in range(half_width, max_row-half_width):
            for col in range(half_width, max_column-half_width):
                if not orig_image[row][col] == 0:
                    # creating neighborhood (based on struct_ID given)
                    neighborhood = []
                    for neigh_index in range(struct_width):
                        width = half_width-neigh_index
                        neighborhood.append(conn_comp[row + width][col-half_width: col+half_width+1])
                    neighborhood = numpy.array(neighborhood)

                    # find list of nonzero values sorted
                    nonzero_neigh = Morphological_Spatial_Filters.list_nonzero(neighborhood, struct_ID)

                    # checking at least 1 value exists in list. 
                    if len(nonzero_neigh) > 0:
                        conn_comp[row][col] = nonzero_neigh[0]
                    else:
                        # no value exists, so creating new value and adding to list of nonzero
                        conn_comp[row][col] = pixel_id
                        nonzero_neigh = [pixel_id]
                        pixel_id += 1

                    # adding current conn_comp ID to equivalence table
                    # if id_index == -1, then it doesn't exist in table, so new entry is created
                    id_index = Morphological_Spatial_Filters.index_of(table, conn_comp[row][col])
                    if id_index == -1:
                        table.append(nonzero_neigh)
                    else:
                        table[id_index] = Morphological_Spatial_Filters.discrete_append(table[id_index], nonzero_neigh)
        # first pass done
        # condense table by merging rows that share values
        table = Morphological_Spatial_Filters.condense_table(table)

        # creating space for new image that highlights components
        components_image = numpy.zeros(orig_image.shape, dtype = "uint8")

        # creating frequency table
        frequency = [0 for num in table]

        # second pass start
        # reassign id to lowest equivalent id
        for row in range(half_width, max_row-half_width):
            for col in range(half_width, max_column-half_width):
                if not conn_comp[row][col] == 0:
                    # finding lowest equivelent ID for current pixel
                    table_index = Morphological_Spatial_Filters.index_of(table, conn_comp[row][col])

                    # creating unique color for each component and populating image array
                    pixel_value = 200*(table_index/len(table)) + 55
                    components_image[row][col] = pixel_value

                    # renaming connected_components so to lowest equivalent ID
                    conn_comp[row][col] = table[table_index][0]

                    # populating frequence table
                    frequency[table_index] += 1

        return components_image, conn_comp, table, frequency
    # End of definition connect_comp
    
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

    def list_nonzero(matrix: numpy.ndarray, struct_id) -> list:
        """
            Returns the lowest nonzero number from a square matrix
        """
        # checking that at least 1 nonzero exists
        max_num = numpy.max(matrix)
        if max_num == 0:
            return []
        
        # bringing current structure as local variable
        c_struct = Morphological_Spatial_Filters.STRUCT_ELEMENT[struct_id]
            
        # getting dimensions
        max_row, max_col = matrix.shape
        
        nonzero = []
        for row in range(max_row):
            for col in range(max_col):
                if not matrix[row][col] == 0:
                    if c_struct[row][col] == 1:
                        nonzero.append(matrix[row][col])
        nonzero.sort()
        return nonzero
    # End of function list_nonzer

    def discrete_append(base_array, append_array):
        """
        
        """
        for num in append_array:
            if not num in base_array:
                base_array.append(num)
        
        return base_array
    # End of definition discrete_append

    def index_of(table, target):
        """
        
        """
        if len(table) == 0:
            return -1
        for index, array in enumerate(table):
            for num in array:
                if num == target:
                    return index
        return -1
    # End of definition index_of

    def condense_table(table):
        indexes_to_delete = []
        for index, array in enumerate(table):
            for num in array:
                lowest_index = Morphological_Spatial_Filters.index_of(table, num)
                if lowest_index == -1:
                    print("condense_table_error:" + str(num))
                if not lowest_index == index:
                    table[lowest_index] = Morphological_Spatial_Filters.discrete_append(table[lowest_index], table[index])
                    if not index in indexes_to_delete:
                        indexes_to_delete.append(index)

        indexes_to_delete.sort()
        for del_index in reversed(indexes_to_delete):
            del table[del_index]

        return table
    # end of condense_table
# End of class Morphological_Spatial_Filters