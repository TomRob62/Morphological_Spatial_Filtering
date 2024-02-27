# Thomas Roberts
# CS 4732: Machine Vision Section 01
# Professor Karakaya
# February 1, 2024

# copied from Project Assignment 2

import numpy
import math


class ImageEnhancement:
    """
    Class Description
    -----------------
    This class focuses on image enhancement algorithms

    Functions
    ---------
    log_transformation
        A method to increase the brightness of an image array using the log function
    power_transformation
        A method to increase the brightness of an image array using the power function
    histogram_equalization
        Method to equalize a grayscale image based on the histogram.
    rgb_histogram_equalization
        Method to equalize a RGB image based on the histogram.
    hsi_histogram_equalization
        Method to equalize a hsi image based on the histogram.
    median_spatial_filter
        Method to reduce noise in an image array by using a median spatial filter 
    mean_spatial_filter
        Method to reduce noise in an image array by using a mean (average) spatial filter
    pad_image
        Method to pad an image array with pixels similar to the nearest edge pixel
    convert_rgb_to_hsi
        Method to convert rgb image model to hsi image model
    convert_hsi_to_rgb
        Method to convert hsi image model to rgb image model
    convert_to_uint8
        Converts an image array of data type 'float' to a data type of 'uint8'
    """

    def log_transformation(image_array: numpy.ndarray) -> numpy.ndarray:
        """
        A method to increase the brightness of an image array using the log function:

            T(r) = c * log(1 + r)

            Where c = 255/(log1+r.max)
            and r = pixel values between 0 and 255

            If the image array data type is 32float or 64float, it will be converted to 
            uint8 because the function only works on values above 1

        Parameters
        ----------
        image_array: NDarray
            original image array
        Coefficient c: float = .50 (optional)
            a coefficient to change the rate of transformation

        Returns
        -------
        NDArray
            a copy of the image array with the applied transformation

        """
        # getting uint8 equivalent if needed
        if (type(image_array[0][0]) == numpy.float32 or type(image_array[0][0]) == numpy.float64):
            image_array = ImageEnhancement.convert_to_uint8(image_array)

        # getting the image dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating copy array to store new pixel values
        array_copy = numpy.zeros((max_row, max_column), dtype="uint8")

        # creating coefficient c
        c = 255 / (math.log(1+image_array.max(), 10))

        # applying transformation
        for row in range(0, max_row):
            for column in range(0, max_column):
                array_copy[row][column] = c * (math.log(1+image_array[row][column], 10))

        return array_copy
    # end of log transormation

    def power_transformation(gamma: float, image_array: numpy.ndarray) -> numpy.ndarray:
        """
        A method to increase the brightness of an image array using the power function:

            T(r) = c * (r**gamma)

            Where c = 1

            This method operates on floating point images. This means: to increase 
            brightness, gamma < 1. To decrease brightness, gamma > 1. 

        Parameters
        ----------
        image_array: NDarray
            original image array
        gamma: int
            The power value

        Returns
        -------
        NDArray
            a copy of the image array with the applied transformation

        """
        # getting the image dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating copy array to store new pixel values
        array_copy = numpy.zeros((max_row, max_column))

        # creating coefficient c
        c = 1

        # applying transformation
        for row in range(0, max_row):
            for column in range(0, max_column):
                array_copy[row][column] = c * \
                    (math.pow(image_array[row][column], gamma))

        return array_copy
    # end of definition power transormation

    def historgram_equalization(image_array: numpy.ndarray) -> numpy.ndarray:
        """
            Method to equalize a grayscale image based on the histogram.

            A step by step process:
            step 1 = finding hist(r)
            step 2 = calculate p(r) 
                    p(r) = sum of (n of j)/n from 0 to k
            step 3 = calculate s 
                    s = sum of p(r) from 0 to k
            step 4: create new array where pixel values are mapped to new values in s

            Paramaters
            ----------
            image_array
                original image array

            Returns
            -------
            NDArray
                equalized image array
        """
        
        # finding hist(r) | step 1
        histogram, bin_edges = numpy.histogram(image_array, bins=256, range=(0, 1))

        # calculating p(r) where p(r) = sum(nj/n) | step 2
        p_of_r = numpy.zeros(256)
        sum_n = histogram.sum() # sum_n = n
        for i in range(256):
            p_of_r[i] = histogram[i]/sum_n # historgram[i] = nj
        
        # calculating s where s = ((previous s) + p(r)) | step 3
        s = numpy.zeros(256)
        cumulative = 0
        for i in range(256):
            cumulative += p_of_r[i]
            s[i] = cumulative
        
        # At this point s represents the equalization (EQ) function
        # Now we apply the EQ function to the original array to produce enhanced image array
            
        # getting dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating space for new image
        array_copy = numpy.zeros(image_array.shape)
        
        # applying EQ function
        for row in range(max_row):
            for column in range(max_column):
                # mapping input r to output s
                index = 255*image_array[row][column] # r
                array_copy[row][column] = s[int(index)] # s
                if(array_copy[row][column] > 1):
                    array_copy[row][column] = 1
                
        return array_copy
    # end of definition

    def rgb_histogram_equalization(image_array: numpy.ndarray) -> numpy.ndarray:
        """
            Method to equalize a RGB image based on the histogram.

                This method splits the rgb into 3 seperate arrays. It then
                calls the histogram_equalition method each of them. Finally,
                it appends the array back together and returns it.

            Paramaters
            ----------
            image_array
                original image array

            Returns
            -------
                equalized image array
        """
        # splitting into 3 arrays
        split_image_array = numpy.split(image_array, 3, axis=2)

        # calling equalization method
        red_equalized = ImageEnhancement.historgram_equalization(split_image_array[0])
        green_equalized = ImageEnhancement.historgram_equalization(split_image_array[1])
        blue_equalized = ImageEnhancement.historgram_equalization(split_image_array[2])

        # combining back into 1 array
        red_green_combined = numpy.append(red_equalized, green_equalized, 2)
        rgb_combined = numpy.append(red_green_combined, blue_equalized, 2)

        return rgb_combined
    # end definition

    def hsi_histogram_equalization(image_array: numpy.ndarray)->numpy.ndarray:
        """
            Method to equalize a hsi image based on the histogram.

                

            Paramaters
            ----------
            image_array
                original image array

            Returns
            -------
                equalized image array
        """
        # sending rgb array to hsi model
        hsi_copy = ImageEnhancement.convert_rgb_to_hsi(image_array)

        # splitting into 3 arrays
        split_image_array = numpy.split(hsi_copy, 3, axis=2)

        # equalizing just h 
        to_float = split_image_array[2]/256
        i = ImageEnhancement.historgram_equalization(to_float)

        # combining hsi arrays
        hs = numpy.append(split_image_array[0], split_image_array[1], 2)
        equalized_hsi = numpy.append(hs, i, 2)

        rgb = ImageEnhancement.convert_hsi_to_rgb(equalized_hsi)

        return rgb
    # end of definition
    
    def mean_spatial_filter(grid_length: int, image_array: numpy.ndarray) -> numpy.ndarray:
        """
            Method to reduce noise in an image array by using a spatial filter to
            average pixel values in a square grid neighborhood.

            If grid_length = 3, then neighborhood dimensions would be 3x3

            Parameters
            ----------
            grid_length: int
                The dimensions of the neighborhood. Must be an odd number
            image_array: NDArray
                image array to be altered  

            Returns
            -------
            NDArray
                A copy array with reduced noise  
        """
        # getting dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating space for new image
        array_copy = numpy.zeros((max_row, max_column), dtype=type(image_array[0][0]))

        # adjusting grid_length if necessary
        if grid_length % 2 == 0:
            grid_length += 1

        # padding image
        padding_length = grid_length//2
        padded_copy = ImageEnhancement.pad_image(padding_length, image_array)

        # applying filter
        for row in range(0, max_row):
            for column in range(0, max_column):
                # averaging neighborhood values
                sum = 0
                for neighbor_row in range(row-padding_length, row+padding_length+1):
                    for neighbor_column in range(column-padding_length, column+padding_length+1):
                        sum += padded_copy[neighbor_row][neighbor_column]

                # end of inner neighborhood for-loops
                average = sum/(grid_length**2)
                array_copy[row][column] = average

         # end of outer image for-loops
        return array_copy
    # end of defintion mean_spatial_filter

    def median_spatial_filter(grid_length: int, image_array: numpy.ndarray) -> numpy.ndarray:
        """
            Method to reduce noise in an image array by using a spatial filter to
            replace current pixel values with the median value in a square grid neighborhood.

            If grid_length = 3, then neighborhood dimensions would be 3x3

            Parameters
            ----------
            grid_length: int
                The dimensions of the neighborhood. Must be an odd number
            image_array: NDArray
                image array to be altered  

            Returns
            -------
            NDArray
                A copy array with reduced noise  
        """
        # getting dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating space for new image
        array_copy = numpy.zeros((max_row, max_column), dtype=type(image_array[0][0]))

        # adjusting grid_length if necessary
        if grid_length % 2 == 0:
            grid_length += 1

        # padding image
        padding_length = grid_length//2
        padded_copy = ImageEnhancement.pad_image(padding_length, image_array)

        # applying filter
        for row in range(0, max_row):
            for column in range(0, max_column):
                # creating list of neighborhood values
                list_values = list()
                for neighbor_row in range(row-padding_length, row+padding_length+1):
                    for neighbor_column in range(column-padding_length, column+padding_length+1):
                        list_values.append(padded_copy[neighbor_row][neighbor_column])

                # finding the median value
                list_values.sort()
                middle_index = len(list_values)//2
                array_copy[row][column] = list_values[middle_index]

         # end of outer image for-loops
        return array_copy
    # end of defintion mean_spatial_filter

    def pad_image(padding_length: int, image_array: numpy.ndarray, ) -> numpy.ndarray:
        """
            Method to pad an image array with pixels similar to the nearest edge pixel

                if image_array size = m*n, and padding_length = 1
                then the new image array will be size (m+2)*(n+2)

            Paramaters
            ----------
            padding_length: int
                the number of pixels of padding to add 
            image_array: NDArray
                image array to be padded

            Returns
            -------
            NDArray
                padded copy of image array
        """
        # getting dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating space for new image
        padded_copy = numpy.zeros((max_row+(2*padding_length),
                                  max_column+(2*padding_length)),
                                  dtype=type(image_array[0][0]))

        # copying orginal values
        for row in range(padding_length, max_row+padding_length):
            for column in range(padding_length, max_column+padding_length):
                padded_copy[row][column] = image_array[row-padding_length][column-padding_length]

        # filling empty rows with edge values
        for row in range(0, max_row+padding_length):
            for column in range(0, padding_length):
                padded_copy[row][column] = padded_copy[row][padding_length]
                padded_copy[row][column+max_column] = padded_copy[row][max_column-padding_length]

        # filling empty columns with edge values
        for column in range(0, max_column+padding_length):
            for row in range(0, padding_length):
                padded_copy[row][column] = padded_copy[padding_length][column]
                padded_copy[row+max_row][column] = padded_copy[max_row-padding_length][column]

        return padded_copy
    # end of definition pad_image

    def convert_rgb_to_hsi(image_array: numpy.ndarray) -> numpy.ndarray:
        """
            A method to convert rgb image model to hsi image model

            h = acos((.5((r-g)+(r-b))/((r-g)**2 + (r-b)(r-b))**.5)
            s = 1 - ((3*min(r,g,b) / r+g+b)
            i = (r+g+b)/3

        Paramaters
        ----------
        image_array
            rgb image array of dtype = float

        Returns
        --------
        NDArray
            hsi image array of dtype = float
        """
        # getting dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        #converting to uint8
        uint8_copy: numpy.ndarray = ImageEnhancement.convert_to_uint8(image_array)

        # creating new array
        array_copy = numpy.zeros(image_array.shape)

        for row in range(max_row):
            for column in range(max_column):
                pixel = numpy.zeros(3)
                r = uint8_copy[row][column][0].astype("float")
                g = uint8_copy[row][column][1].astype("float")
                b = uint8_copy[row][column][2].astype("float")

                numerator = .5*((r-g) + (r-b))
                denominator = math.sqrt((r-g)**2 + (r-b)*(g-b))
                
                if denominator == 0:
                    theta = 0
                else:
                    theta = numerator/denominator

                theta = theta * (math.pi/180)
                theta = math.acos(theta)
                h = theta
                if(b > g):
                    h = (2*math.pi)-h

                s = 1 - ((3*numpy.min(uint8_copy[row][column]))/(r+b+g))
                i = (r+b+g)/3

                pixel[0] = h
                pixel[1] = s
                pixel[2] = i

                array_copy[row][column] = pixel

        return array_copy
    # end of definition convert_rgb_to_hsi

    def convert_hsi_to_rgb(image_array: numpy.ndarray) -> numpy.ndarray:
        """
            A method to convert hsi image model to rgb image model

            Formula were copied from class ppt. Lecture 3 - visual perception
            slide 64

        Paramaters
        ----------
        image_array
            hsi image array of dtype = float

        Returns
        --------
        NDArray
            rgb image array of dtype = uint8
        """
        # getting dimensions
        max_row = len(image_array)
        max_column = len(image_array[0])

        # creating new array
        array_copy = numpy.zeros(image_array.shape, dtype="uint8")

        # applying formula from class ppt.
        for row in range(max_row):
            for column in range(max_column):
                h = image_array[row][column][0]
                s = image_array[row][column][1]
                i = image_array[row][column][2]
                pixel = numpy.zeros(3)
                if h < .67:
                    pixel[0] = i*(1+((s*math.cos(h))/(math.cos(.33-h))))
                    pixel[2] = i*(1-s)
                    pixel[1] = 3*i - (pixel[0] + pixel[2])
                elif h < 1.33:
                    pixel[0] = i*(1-s)
                    pixel[1] = i*(1+((s*math.cos(h-.67))/(math.cos(h-.33))))
                    pixel[2] = 3*i - (pixel[0] + pixel[1])
                else:
                    pixel[1] = i*(1-s)
                    pixel[2] = i*(1+((s*math.cos(h-1.33))/(math.cos(1.67-h))))
                    pixel[0] = 3*i - (pixel[1] + pixel[2])
                pixel = pixel*256
                pixel = pixel.astype("uint8")
                array_copy[row][column] = pixel
        return array_copy
    # end of definition
        
    def convert_to_uint8(image_array: numpy.ndarray):
        """
        Converts an image array of data type 'float' to a data type of 'uint8'

        Paramaters
        ----------
        image_array: NDArray
            original image array

        Returns
        -------
        NDArray
            of dtype = numpy.uint8
        """
        # finding row/column size
        max_row = len(image_array)
        max_column = len(image_array[0])

        # finding min/max values for normalization
        max_pixel_val = image_array.max()
        min_pixel_val = image_array.min()

        # creating space for copy
        array_copy = numpy.zeros(image_array.shape, dtype= 'uint8')

        # converting each pixel
        for row in range(0, max_row):
            for column in range(0, max_column):
                array_copy[row][column] = 255*image_array[row][column]
        # ((image_array[row][column]-min_pixel_val)/(max_pixel_val-min_pixel_val))
        # end for loops
        return array_copy
    # end definition convert to uint8
