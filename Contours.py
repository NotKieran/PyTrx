# Import packages
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import math
import matplotlib.lines as lines
import cv2

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io
from skimage.color import rgb2gray
from cycler import cycler

# Import PyTrx packages
import Utilities
import Line
import FileHandler
import Velocity
import CamEnv

from Line import getOGRLine
from CamEnv import CamEnv
from CamEnv import projectUV

sys.path.append('../')

def write_snake(snake, image_name, destination):
    f = open(destination + ".txt", 'w+')
    f.write(image_name)
    f.write('\n')
    f.write(repr(snake))


def open_image(image_path):
    return FileHandler.readImg(image_path)


def line_from_coords(pxcoords):
    xy = pxcoords[0]
    x = xy[:, 0]
    y = xy[:, -1]

    return np.array([x, y]).T


def plot_snakes(snakes, image, start_line, title, initial_label="Initial Line"):
    plt.rc('axes', prop_cycle=(cycler('color', ['b', 'm', 'r', 'y', 'c', 'g']) +
                               cycler('linestyle', ['-', '-', '-', '-', '-', '-'])))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(start_line[:, 0], start_line[:, 1], lw=1, label=initial_label)

    for index, snake in enumerate(snakes):
        ax.plot(snake[:, 0], snake[:, 1], lw=1, label="Snake #" + str(index))

    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, image.shape[1], image.shape[0], 0])
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_snakes(snakes, image, start_line, title, destination, display=True, failed=False,
                initial_label="Initial Line", scale=1):
    start_line = scale_line(start_line, scale)

    if failed:
        plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r','y']) +
                                   cycler('linestyle', ['-', '--', '-.'])))
    else:
        plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g','y']) +
                                   cycler('linestyle', ['-', '-', '-.'])))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(start_line[:, 0], start_line[:, 1], lw=1, label=initial_label)

    for index, snake in enumerate(snakes):
        snake = scale_line(snake, scale)
        ax.plot(snake[:, 0], snake[:, 1], lw=1, label="Snake #" + str(index))

    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, image.shape[1], image.shape[0], 0])
    ax.legend()
    plt.title(title)

    destination += title
    if os.path.exists(str(destination) + '.png'):
        plt.savefig(str(destination) + '_{}.png'.format(int(time.time())), dpi=300)
    else:
        plt.savefig(str(destination) + '.png', dpi=300)

    plt.tight_layout()

    if display:
        plt.show()

    plt.close('all')


def save_line(line, Imn, path):
    if os.path.exists('line.png'):
        path += 'line_{}.png'.format(int(time.time()))
    else:
        path += 'line.png'

    FileHandler.writeLineCoords(line, None, Imn, path, None)


def moved_right(true_line, new_line, relaxation):
    movement_array = []
    relaxation = relaxation / 100
    relaxation += 1
    for index, point in enumerate(true_line):
        if true_line[index][0] * relaxation < new_line[index][0]:
            # If old X < new X, line has moved right
            movement_array.append(True)
        else:
            movement_array.append(False)

    return movement_array


def has_moved_right(true_line, new_line, relaxation):
    movement_array = moved_right(true_line, new_line, relaxation)
    for point in movement_array:
        if point == True:
            return True
    return False


def get_curve_difference(previous_line, current_line):
    distance_array = []
    for index, point in enumerate(previous_line):
        x = abs(previous_line[index][0] - current_line[index][0])
        y = abs(previous_line[index][1] - current_line[index][1])
        distance_array.append(math.sqrt(x**2 + y**2))
    return distance_array


def get_max_curve_difference(previous_line, current_line):
    return max(get_curve_difference(previous_line, current_line))


def get_total_curve_difference(previous_line, current_line):
    return sum(get_curve_difference(previous_line, current_line))


def scale_image(img, ratio):
    height, width = img.shape
    w_size = int(width * ratio)
    h_size = int(height * ratio)
    return cv2.resize(img, (w_size, h_size), interpolation=cv2.INTER_AREA)


def scale_line(line, ratio):
    reduced_line = []
    for point in line:
        x = int(ratio * point[0])
        y = int(ratio * point[1])
        reduced_line.append(np.array([x, y]))
    return np.array(reduced_line)


def create_image_title(image_name, day, initial=False):
    day = str(day)
    if initial:
        return "Day " + day + " - " + image_name + " - " + initial
    else:
        return "Day " + day + " - " + image_name


def create_folder_name(ends, alpha, beta, w_line, w_edge, gamma, convergence, start_from_initial, scale, sigma):
    name = "_{}_a-{}_b-{}_line-{}_edge-{}_g-{}_con-{}_use-initial-line-{}_scale-{}_sigma-{}/"
    return name.format(ends, alpha, beta, w_line, w_edge, gamma, convergence, start_from_initial, scale, sigma)


def failed_snake(initial, snake, relaxation, difference):
    moved_right = False
    too_different = False

    if has_moved_right(initial, snake, relaxation):
        moved_right = True

    if (get_max_curve_difference(initial, snake) > difference):
        too_different = True

    if (too_different or moved_right):
        return True
    else:
        return False


def setup_image(image, scale, sigma):
    gray_image = rgb2gray(image)
    scaled_image = scale_image(gray_image, scale)
    filtered_image = gaussian(scaled_image, sigma, multichannel=False)
    return filtered_image


def future_confidence(present_image_name, initial_contour, present_snake, camimgs, cam, invmask, hgback,
                      hgmax, hgqual, hgmind, hgminf, alpha, beta, w_line,
                      w_edge, gamma, ends, convergence, scale, sigma, relaxation, difference):
    future_image_number = int(filter(str.isdigit, present_image_name)) + 48
    future_image_name = "IMG_{}.JPG".format(future_image_number)

    future_image = load_image(future_image_name, camimgs, cam, invmask, hgback, hgmax, hgqual, hgmind, hgminf, scale,
                              sigma)

    if future_image[0][0] != False:
        future_snake = active_contour(snake=initial_contour, image=future_image, alpha=alpha, beta=beta, w_line=w_line,
                        w_edge=w_edge, gamma=gamma, bc=ends, convergence=convergence)
        if not failed_snake(initial_contour, future_snake, relaxation, 5000):
            return [failed_snake(present_snake, future_snake, relaxation, 5000), future_snake]
        else:
            return [False, [[-1, -1]]]
    else:
        return [False,[[-1,-1]]]


def load_image(image_name, camimgs, cam, invmask, hgback, hgmax, hgqual, hgmind, hgminf, scale, sigma):
    try:
        camimgs += image_name
        # Set up Homography object
        homog = Velocity.Homography(camimgs, cam, invmask, calibFlag=True, band='L', equal=True)
        # Calculate homography
        hg = homog.calcHomographyPairs(hgback, hgmax, hgqual, hgmind, hgminf)
        homogmatrix = [item[0] for item in hg]
        new_terminus = Line(camimgs, cam, homogmatrix)
        current_imgset = new_terminus._imageSet
        imn = new_terminus.getImageNames()

        cameraMatrix = cam.getCamMatrixCV2()
        distortP = cam.getDistortCoeffsCV2()    

        # Create New Snake
        image = current_imgset[0].getImageCorr(cameraMatrix, distortP)
        processed_image = setup_image(image, scale, sigma)
        return processed_image
    except:
        return [[False]]


def calcUVXYZ(pxpts, hmatrix, camEnv):
    # xyzline (list):         Line length (xyz)
    # xyzpts (list):          XYZ coordinates of lines
    # pxline (list):          Line length (px)
    # pxpts (list):           UV coordinates of lines

    invprojvars = CamEnv.setProjection(camEnv.getDEM(), camEnv._camloc,
                                       camEnv._camDirection,
                                       camEnv._radCorr,
                                       camEnv._tanCorr,
                                       camEnv._focLen,
                                       camEnv._camCen,
                                       camEnv._refImage)

    # Calculate homography-corrected pts if desired
    # if hmatrix is not None:
    #     print 'Correcting for camera motion'
    #     pxpts = Velocity.apply_persp_homographyPts(pxpts, hmatrix, inverse=True)

    # Re-format pixel point coordinates
    pxpts = np.squeeze(pxpts)

    # Create OGR pixl line object and extract length
    pxline = getOGRLine(pxpts)
    print 'Line contains %i points' % (pxline.GetPointCount())
    pxline = pxline.Length()
    print 'Line length: %d px' % (pxline)

    if invprojvars is not None:
        # Get xyz coordinates with inverse projection
        xyzpts = projectUV(pxpts, invprojvars)

        # Create ogr line object
        xyzline = getOGRLine(xyzpts)
        xyzline = xyzline.Length()

        print 'Line length: %d m' % (xyzline)

        return [[xyzline, xyzpts], [pxline, pxpts]]

    else:
        # Return pixel coordinates only
        return [[None, None], [pxline, pxpts]]


def save_xyz_uv(input_line,hmatrix,camEnv,name,destination):
    lines = calcUVXYZ(input_line,hmatrix,camEnv)
    pxcoords = lines[1][1]
    xyzcoords = lines[0][1]

    xyz_name = name + "xyzcoords.txt"
    uv_name = name + "uvcoords.txt"
    # Write line coordinates to txt file
    FileHandler.writeLineCoords([pxcoords.tolist()], [xyzcoords.tolist()], name,
                                destination + uv_name,
                                destination + xyz_name)

