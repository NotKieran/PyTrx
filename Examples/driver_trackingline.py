# Import packages
import Utilities
import Contours
import FileHandler
from CamEnv import CamEnv
from Velocity import Homography
from Line import Line
import os
import sys
import time

from skimage.color import rgb2gray
from skimage.segmentation import active_contour
from skimage.filters import gaussian

# Import PyTrx packages
sys.path.append('../')
# ---------------------------   Initialisation   -------------------------------

# Define data input directories
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR1_2014.txt'
invmask = '../Examples/camenv_data/invmasks/KR1_2014_inv.jpg'
# Main Image Folder
camimgs = '../Examples/images/Kronebreen/'
# Initial Image
initial_image = 'IMG_3038.JPG'
camimgs_start = camimgs + initial_image
# Define data output directory
destination = '../Examples/results/trackingline/'

if not os.path.exists(destination):
    os.makedirs(destination)
# Define camera environment
cam = CamEnv(camdata)

# ---------------------   Calculate homography   -------------------------------
# Set homography parameters
hgback = 1.0  # Back-tracking threshold
hgmax = 50000  # Maximum number of points to seed
hgqual = 0.1  # Corner quality for seeding
hgmind = 5.0  # Minimum distance between seeded points
hgminf = 4  # Minimum number of seeded points to track

# Set up Homography object
homog = Homography(camimgs_start, cam, invmask,
                   calibFlag=True, band='L', equal=True)

# Calculate homography
hg = homog.calcHomographyPairs(hgback, hgmax, hgqual, hgmind, hgminf)
homogmatrix = [item[0] for item in hg]
# -----------------------   Calculate/import lines   ---------------------------
# Set up line object
terminus = Line(camimgs_start, cam, homogmatrix)

# Used to select a file to start the tracking from
start_frame = 'ReportImages/tests/'  # Path from destination
start_frame_name = 'IMG_3211.JPG'  # File name
start_frame += start_frame_name

xyzfile = destination + start_frame + 'xyzcoords.txt'
pxfile = destination + start_frame + 'uvcoords.txt'
lines = FileHandler.importLineData(xyzfile, pxfile)
# Manually define terminus lines
# lines = terminus.calcManualLines()
# ----------------------------   Export data   ---------------------------------

# Get image names and line data
imn = terminus.getImageNames()
pxcoords = [item[1][1] for item in lines]
xyzcoords = [item[0][1] for item in lines]

# Print start frame map, used for Design 4 artifact creation
Utilities.plotLineXYZ(xyzcoords[0], cam.getDEM(
), show=False, save=destination + 'Tracked_xyz_' + start_frame_name)


# Write line coordinates to txt file
# FileHandler.writeLineCoords(pxcoords, xyzcoords, imn,
#                             destination + 'TESTuvcoord.txt',
#                             destination + 'TESTxyzcoords.txt')

# -----------------------   Create Contour From Line   -------------------------
line = Contours.line_from_coords(pxcoords)
cameraMatrix = cam.getCamMatrixCV2()
distortP = cam.getDistortCoeffsCV2()

current_imgset = terminus._imageSet
image = current_imgset[0].getImageCorr(cameraMatrix, distortP)

# -----------------------   Set Snake Parameters   -------------------------
ends = 'free'
alpha = 0.01        # Lower than 1 is ideal
beta = 0.5          # Lower than 1 is ideal
w_line = -10        # Must be negative
w_edge = 20         # Must be positive
gamma = 0.01        # Time Stepping, no reason to change
convergence = 0.01  # Lower than 1 is ideal

scale = 0.1  # Image scaling, 0.1 works well
sigma = 1           # Filter Size, 1 is default

# Start from initial contour for the series or use previous image's snake
start_from_initial = False

display = False     # Display the images when created

use_future = False  # Use future comparison in convergence
use_failure = True  # Use fault detection

relaxation = 5      # Relaxation parameter, used as a percentage for rightwards motion
difference = 150    # Total pixel distance allowed for each point in the snake

name = Contours.create_folder_name(ends, alpha, beta, w_line, w_edge, gamma, convergence, start_from_initial,
                                   scale, sigma)

# Path to results location

destination += "ReportImages/"
# destination += "Kronebreen_set" + name
destination += "tests/"

if os.path.exists(destination):
    destination += "{}".format(int(time.time())) + "/"
    os.makedirs(destination)
else:
    os.makedirs(destination)

# Save snake parameters to file for reference
filename = destination + "param.txt"
f = open(filename, "w+")
f.write(name)
f.close()

processed_image = Contours.setup_image(image, scale, sigma)
scaled_line = Contours.scale_line(line, scale)

snake = active_contour(snake=scaled_line, image=processed_image, alpha=alpha, beta=beta, w_line=w_line,
                       w_edge=w_edge, gamma=gamma, bc=ends, convergence=convergence)

failed = Contours.failed_snake(line, Contours.scale_line(
    snake, 1 / scale), relaxation, difference)
Contours.save_snakes([snake], image, scaled_line, "Snake_Initialisation", destination, display=display, failed=failed,
                     scale=(1 / scale))

# Call terminus again with full array of images minus the first, iterate through them plotting every image with
# snake and previous image
image_set = []

for file_object in os.listdir(camimgs):
    if file_object.endswith(".JPG"):
        image_set.append(camimgs + file_object)

image_set.sort()

if start_from_initial:
    previous_line = scaled_line
else:
    previous_line = snake

day = 1
for index, image in enumerate(image_set):
    # Check Image Day
    if (index % 48 == 0) and (index != 0):
        day += 1

    # Set up Homography object
    homog = Homography(image, cam, invmask, calibFlag=True,
                       band='L', equal=True)
    # Calculate homography
    hg = homog.calcHomographyPairs(hgback, hgmax, hgqual, hgmind, hgminf)
    homogmatrix = [item[0] for item in hg]
    new_terminus = Line(image, cam, homogmatrix)
    current_imgset = new_terminus._imageSet
    imn = new_terminus.getImageNames()

    # Create New Snake
    image = current_imgset[0].getImageCorr(cameraMatrix, distortP)
    processed_image = Contours.setup_image(image, scale, sigma)
    snake = active_contour(snake=previous_line, image=processed_image, alpha=alpha, beta=beta, w_line=w_line,
                           w_edge=w_edge, gamma=gamma, convergence=convergence, bc=ends)

    # Save Image and prepare next snake
    title = Contours.create_image_title(str(imn[0]), day)
    snake_array = [snake]
    failed = False

    if use_failure == True:
        failed = Contours.failed_snake(line, Contours.scale_line(
            snake, 1 / scale), relaxation, difference)

    if use_future == True:
        future_confidence = Contours.future_confidence(str(imn[0]), previous_line, snake, camimgs, cam, invmask, hgback, hgmax,
                                                       hgqual, hgmind, hgminf, alpha, beta, w_line, w_edge, gamma, ends, convergence, scale, sigma, relaxation, difference)
        failed = (future_confidence[0] or failed)
        future_snake = future_confidence[1]
        snake_array.append(future_snake)

    if start_from_initial:
        Contours.save_snakes(snake_array, image, previous_line,
                             title, destination, display, failed, scale=(1 / scale))
    else:
        Contours.save_snakes(snake_array, image, previous_line, title, destination,
                             display, failed,  initial_label="Previous Snake", scale=(1 / scale))

        if not failed:
            previous_line = snake

    Contours.save_xyz_uv(Contours.scale_line(
        snake, 1 / scale), homogmatrix, cam, str(imn[0]), destination)


print '\n\nFinished'
