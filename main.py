# Thomas Roberts
# CS 4732 Section 01: Machine Vision
# Professor Karakaya
# February 23, 2024
#
# last modified February 23, 2024


from matplotlib import pyplot as plt
from filters import Morphological_Spatial_Filters as filter
from image_enhancement import ImageEnhancement as ehance
import numpy

# 1) Sharpen the edge of moon image (moon.jpg) by using the Laplace and Sobel filters
orig_image = plt.imread("images\\moon.jpg")

# laplace
laplace_moon_image = filter.laplace_filter(orig_image)
file_name = "results\\laplace_filter_moon.jpg"
plt.imsave(file_name, laplace_moon_image, cmap = "gray")

# Sobel
for i in range(0, 10, 2):
    coeff = i/10
    sobel_moon_image = filter.sobel_filter(orig_image, coeff)
    file_name = "results\\sobel_filter_" + str(i) + "_moon.jpg"
    plt.imsave(file_name, sobel_moon_image, cmap = "gray")

# 2) clean the noises and remove the holes in the image as much as 
# possible by using different size of structuring elements
orig_image = plt.imread("images\\fingerprint.jpg")

# histogram
histogram, bin_edges = numpy.histogram(orig_image, bins=256)
plt.figure()
plt.xlim((0, 255))
plt.plot(bin_edges[0:-1], histogram)
file_name = "results\\histogram_fingerprint.jpg"
plt.savefig(file_name)

# filer noise
orig_image = ehance.median_spatial_filter(3, orig_image)
file_name = "results\\noise_filter_fingerprint.jpg"
plt.imsave(file_name, orig_image, cmap = "gray")

# convert tp binary
binary_image = filter.binary(orig_image)
binary_image = numpy.invert(binary_image)
file_name = "results\\binary_fingerprint.jpg"
plt.imsave(file_name, binary_image, cmap = "gray")

# erode
eroded_fingerprint = filter.erosion(binary_image, 1)
file_name = "results\\eroded_fingerprint.jpg"
plt.imsave(file_name, eroded_fingerprint, cmap = "gray")

# dialate
dialate_fingerprint = filter.dilution(eroded_fingerprint, 0)
dialate_fingerprint = filter.dilution(dialate_fingerprint, 0)
file_name = "results\\dialated_fingerprint.jpg"
plt.imsave(file_name, dialate_fingerprint, cmap = "gray")

# save final copy
final_fingerprint = filter.erosion(dialate_fingerprint, 1)
file_name = "results\\final_fingerprint.jpg"
plt.imsave(file_name, final_fingerprint, cmap = "gray")


# 3) In cell.jpg,  write a code to count the total number of cells, calculate the size 
# of each cell in pixels, and show the boundary of the biggest cell in an output image
orig_image = plt.imread("images\\cell.jpg")

# count cells
binary_copy = filter.binary(orig_image, 15)
file_name = "results\\binary_cell.jpg"
plt.imsave(file_name, binary_copy, cmap = "gray")

# print cell size
connect_comp_image, pixel_ID, table, freq = filter.connected_comp(binary_copy, 1)
file_name = "results\\connect_comp_cell.jpg"
plt.imsave(file_name, connect_comp_image, cmap = "gray")
print("Number of cells: " + str(len(freq)))
print("\nAverage component size: " + str(numpy.average(freq)))
print(freq)

# boundary of largest cell
highest_freq = max(freq)
index_highest = freq.index(highest_freq)
target_id = table[index_highest][0]
boundary_extract_image = filter.extract_boundary(pixel_ID, target_id, 1)
file_name = "results\\boundary_largest_cell.jpg"
plt.imsave(file_name, boundary_extract_image, cmap = "gray")
