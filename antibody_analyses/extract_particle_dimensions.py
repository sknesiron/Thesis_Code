import csv  
import multiprocessing as mp
import cv2 
import matplotlib.pyplot as plt
import matplotx
import mrcfile
import numpy as np
from matplotlib import patheffects as pe  
from matplotlib.patches import Ellipse  
from matplotlib.widgets import Slider
from scipy.spatial import ConvexHull 
from skimage.filters import (
    butterworth,
 
    threshold_otsu,
)
from skimage.measure import  label, regionprops
from skimage.morphology import (
    dilation,
    disk,
    erosion,
    remove_objects_by_distance,
    remove_small_holes,
)
from tqdm import tqdm


def contrast_normalization(arr_bin, tile_size=128):
    """
    Computes the minimum and maximum contrast values to use
    by calculating the median of the 2nd/98th percentiles
    of the mic split up into tile_size * tile_size patches.
    :param arr_bin: the micrograph represented as a numpy array
    :type arr_bin: list
    :param tile_size: the size of the patch to split the mic by
        (larger is faster)
    :type tile_size: int
    from: https://discuss.cryosparc.com/t/differences-between-2d-ctf-when-viewed-in-cryosparc-vs-exported-ctf-diag-image-path/10511/6
    """
    ny, nx = arr_bin.shape
    # set up start and end indexes to make looping code readable
    tile_start_x = np.arange(0, nx, tile_size)
    tile_end_x = tile_start_x + tile_size
    tile_start_y = np.arange(0, ny, tile_size)
    tile_end_y = tile_start_y + tile_size
    num_tile_x = len(tile_start_x)
    num_tile_y = len(tile_start_y)

    # initialize array that will hold percentiles of all patches
    tile_all_data = np.empty((num_tile_y * num_tile_x, 2), dtype=np.float32)

    index = 0
    for y in range(num_tile_y):
        for x in range(num_tile_x):
            # cut out a patch of the mic
            arr_tile = arr_bin[
                tile_start_y[y] : tile_end_y[y], tile_start_x[x] : tile_end_x[x]
            ]
            # store 2nd and 98th percentile values
            tile_all_data[index:, 0] = np.percentile(arr_tile, 98)
            tile_all_data[index:, 1] = np.percentile(arr_tile, 2)
            index += 1

    # calc median of non-NaN percentile values
    all_tiles_98_median = np.nanmedian(tile_all_data[:, 0])
    all_tiles_2_median = np.nanmedian(tile_all_data[:, 1])
    vmid = 0.5 * (all_tiles_2_median + all_tiles_98_median)
    vrange = abs(all_tiles_2_median - all_tiles_98_median)
    extend = 1.5
    # extend vmin and vmax enough to not include outliers
    vmin = vmid - extend * 0.5 * vrange
    vmax = vmid + extend * 0.5 * vrange

    return vmin, vmax


def butterworth_filter(img):
    """
    Apparently this comes close to what you see in cryosparc jobs when they display particles according to:
    https://discuss.cryosparc.com/t/inspect-raw-images-of-particles-of-certain-2-3d-classes/12261/8
    """
    return butterworth(
        img, cutoff_frequency_ratio=3 / img.shape[0], high_pass=False, order=1
    )


def bin_images(images, bin_factor):
    """
    Applies binning to a series of images by reducing their resolution.

    Parameters:
        images (numpy.ndarray): A 3D array of shape (N, H, W),
                                where N is the number of images,
                                H is the height, and W is the width.
        bin_factor (int): The factor by which to reduce the resolution (e.g., 2 means 2x2 binning).

    Returns:
        numpy.ndarray: A 3D array of binned images with shape (N, H//bin_factor, W//bin_factor).
    """
    if bin_factor <= 0:
        raise ValueError("Bin factor must be a positive integer.")
    if images.ndim != 3:
        raise ValueError("Input images must be a 3D array of shape (N, H, W).")

    # Validate that the dimensions are divisible by bin_factor
    N, H, W = images.shape
    if H % bin_factor != 0 or W % bin_factor != 0:
        raise ValueError("Image dimensions must be divisible by the bin_factor.")

    # Reshape and compute the mean for binning
    binned_images = images.reshape(
        N, H // bin_factor, bin_factor, W // bin_factor, bin_factor
    ).mean(axis=(2, 4))

    return binned_images

def crop_center_square_batch(images, side_length):
    """
    Crops a centered square of given side_length from each image in a batch.

    Parameters:
        images (numpy.ndarray): A 3D array (N, H, W) where N is the number of images.
        side_length (int): The side length of the cropped square.

    Returns:
        numpy.ndarray: A batch of cropped images with shape (N, side_length, side_length).
    """
    side_length = side_length // 2
    center_x = images.shape[1] // 2
    center_y = images.shape[2] // 2

    return images[
        :,  # Keep all images in the batch
        max(center_x - side_length, 0) : min(center_x + side_length, images.shape[1]),
        max(center_y - side_length, 0) : min(center_y + side_length, images.shape[2]),
    ]

def compute_mask(image):
    m = image > threshold_otsu(image)
    m = erosion(m, disk(3))
    m = remove_objects_by_distance(label(m), min_distance=image.shape[0])
    m = dilation(m, disk(3))
    return remove_small_holes(m > 0, area_threshold=100)


def compute_mask_batch(images):
    return np.array([compute_mask(img) for img in tqdm(images, "Computing Masks")])


def get_area(mask, pixel_size):
    return np.sum(mask) * pixel_size**2


def get_area_batch(mask_batch, pixel_size):
    return np.array(
        [get_area(mask, pixel_size) for mask in tqdm(mask_batch, "Area-Calculation")]
    )


def main_convex_hull():
    with mrcfile.open("data/batch_0_restacked.mrc") as mrc:
        data = mrc.data

    init_frame = 0
    crop_size = 150
    bin_fact = 1

    cropped_data = crop_center_square_batch(data, crop_size)
    bin_data = bin_images(cropped_data, bin_fact)
    lp_data = np.zeros(bin_data.shape)

    for i, img in enumerate(tqdm(bin_data, f"Butterworth-Filter")):
        lp_data[i] = butterworth_filter(img)

    clims = [
        contrast_normalization(img) for img in tqdm(lp_data, "Contrast-Normalization")
    ]

    num_workers = mp.cpu_count()
    chunk_size = lp_data.shape[0] // num_workers
    chunks = [
        lp_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)
    ]

    with mp.Pool(num_workers) as pool:
        results = pool.map(compute_mask_batch, chunks)

    masks = np.concatenate(results, axis=0)

    lp_data = lp_data[0 : masks.shape[0]]

    areas = get_area_batch(masks, 0.83)

    regions = [
        max(regionprops(label(binary)), key=lambda r: r.area) for binary in masks
    ]

    ellipse_params = []
    hull_points_list = []
    for region in tqdm(regions):
        yx_coords = region.coords  # skimage gives (row, col) coordinates
        y = yx_coords[:, 0]
        x = yx_coords[:, 1]

        # Compute the convex hull
        points = np.column_stack([x, y])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the loop
        hull_points_list.append(hull_points)

        # Fit an ellipse using OpenCV's fitEllipse
        if len(hull_points) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(hull_points)
            (xc, yc), (major_axis, minor_axis), angle = ellipse
            ellipse_params.append((yc, xc, major_axis, minor_axis, angle))
        else:
            print("Not enough points to fit an ellipse!")
            return

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    image = ax[0].imshow(lp_data[init_frame], cmap="gray_r", clim=clims[init_frame])
    mask = ax[1].imshow(masks[init_frame], cmap="gray_r")
    ellipse = Ellipse(
        (ellipse_params[init_frame][1], ellipse_params[init_frame][0]),
        width=ellipse_params[init_frame][2],
        height=ellipse_params[init_frame][3],
        angle=ellipse_params[init_frame][4],
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    ax[0].add_patch(ellipse)

    # Plot the width and height lines
    xc, yc = ellipse_params[init_frame][1], ellipse_params[init_frame][0]
    major_axis, minor_axis = (
        ellipse_params[init_frame][2],
        ellipse_params[init_frame][3],
    )
    angle = np.deg2rad(ellipse_params[init_frame][4])
    dx_major = major_axis / 2 * np.cos(angle)
    dy_major = major_axis / 2 * np.sin(angle)
    dx_minor = minor_axis / 2 * np.cos(angle + np.pi / 2)
    dy_minor = minor_axis / 2 * np.sin(angle + np.pi / 2)

    (line_major,) = ax[0].plot(
        [xc - dx_major, xc + dx_major],
        [yc - dy_major, yc + dy_major],
        color="blue",
        linestyle="--",
        linewidth=1,
    )
    (line_minor,) = ax[0].plot(
        [xc - dx_minor, xc + dx_minor],
        [yc - dy_minor, yc + dy_minor],
        color="green",
        linestyle="--",
        linewidth=1,
    )

    # Add text annotations for width and height
    text_major = ax[0].text(
        xc + dx_major,
        yc + dy_major,
        f"Width: {major_axis:.2f}",
        color="blue",
        fontsize=8,
        ha="center",
    )
    text_minor = ax[0].text(
        xc + dx_minor,
        yc + dy_minor,
        f"Height: {minor_axis:.2f}",
        color="green",
        fontsize=8,
        ha="center",
    )

    # Plot the convex hull on the mask image
    (hull_line,) = ax[1].plot(
        hull_points_list[init_frame][:, 0],
        hull_points_list[init_frame][:, 1],
        "r-",
        linewidth=1,
    )

    ax[0].set(title=f"Frame {init_frame}")
    ax[1].set(title=f"Mask {areas[init_frame]:.2f} Å²")
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax_slider, "Frame", 0, lp_data.shape[0] - 1, valinit=init_frame, valstep=1
    )
    fig.colorbar(image, ax=ax)

    def update(val):
        frame = int(slider.val)  # Get the current slider value
        image.set(data=lp_data[frame], clim=clims[frame])
        mask.set(data=masks[frame], clim=(0, 1))
        ellipse.set_center((ellipse_params[frame][1], ellipse_params[frame][0]))
        ellipse.width = ellipse_params[frame][2]
        ellipse.height = ellipse_params[frame][3]
        ellipse.angle = ellipse_params[frame][4]

        # Update the width and height lines
        xc, yc = ellipse_params[frame][1], ellipse_params[frame][0]
        major_axis, minor_axis = ellipse_params[frame][2], ellipse_params[frame][3]
        angle = np.deg2rad(ellipse_params[frame][4])
        dx_major = major_axis / 2 * np.cos(angle)
        dy_major = major_axis / 2 * np.sin(angle)
        dx_minor = minor_axis / 2 * np.cos(angle + np.pi / 2)
        dy_minor = minor_axis / 2 * np.sin(angle + np.pi / 2)

        line_major.set(
            data=([xc - dx_major, xc + dx_major], [yc - dy_major, yc + dy_major])
        )
        line_minor.set(
            data=([xc - dx_minor, xc + dx_minor], [yc - dy_minor, yc + dy_minor])
        )

        # Update text annotations for width and height
        text_major.set_position((xc + dx_major, yc + dy_major))
        text_major.set_text(f"Width: {major_axis:.2f}")
        text_minor.set_position((xc + dx_minor, yc + dy_minor))
        text_minor.set_text(f"Height: {minor_axis:.2f}")

        # Update the convex hull on the mask image
        hull_line.set_data(
            hull_points_list[frame][:, 0],
            hull_points_list[frame][:, 1],
        )

        ax[0].set_title(f"Frame {frame}")
        ax[1].set(title=f"Mask {areas[frame]:.2f} Å²")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    plt.close()

    np.save("results/filtered_particles.npy", lp_data)
    np.save("results/masks.npy", masks)

    plt.style.use(matplotx.styles.pitaya_smoothie["light"])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(areas, bins=50)
    ax.set(xlim=(000, 10000), ylabel="Counts", xlabel="Area [Å^2]")
    plt.show()
    plt.close()


def main_mask():
    with mrcfile.open("data/particle_stack.mrc") as mrc:
        data = mrc.data

    init_frame = 0
    crop_size = 150
    bin_fact = 1
    pixel_size = 0.83

    cropped_data = crop_center_square_batch(data, crop_size)
    bin_data = bin_images(cropped_data, bin_fact)
    lp_data = np.zeros(bin_data.shape)

    for i, img in enumerate(tqdm(bin_data, f"Butterworth-Filter")):
        lp_data[i] = butterworth_filter(img)

    clims = [
        contrast_normalization(img) for img in tqdm(lp_data, "Contrast-Normalization")
    ]

    num_workers = mp.cpu_count()
    chunk_size = lp_data.shape[0] // num_workers
    chunks = [
        lp_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)
    ]

    with mp.Pool(num_workers) as pool:
        results = pool.map(compute_mask_batch, chunks)

    masks = np.concatenate(results, axis=0)

    lp_data = lp_data[0 : masks.shape[0]]

    areas = get_area_batch(masks, pixel_size=pixel_size)

    ellipse_params = []
    for mask in tqdm(masks):
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # Fit an ellipse using OpenCV's fitEllipse
            ellipse = cv2.fitEllipse(contours[0])
            (xc, yc), (major_axis, minor_axis), angle = ellipse
            ellipse_params.append((yc, xc, major_axis, minor_axis, angle))
        else:
            print("No contours found to fit an ellipse!")
            return

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    image = ax[0].imshow(lp_data[init_frame], cmap="gray_r", clim=clims[init_frame])
    mask = ax[1].imshow(masks[init_frame], cmap="gray_r")
    ellipse = Ellipse(
        (ellipse_params[init_frame][1], ellipse_params[init_frame][0]),
        width=ellipse_params[init_frame][2],
        height=ellipse_params[init_frame][3],
        angle=ellipse_params[init_frame][4],
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    ax[0].add_patch(ellipse)

    # Plot the width and height lines
    xc, yc = ellipse_params[init_frame][1], ellipse_params[init_frame][0]
    major_axis, minor_axis = (
        ellipse_params[init_frame][2],
        ellipse_params[init_frame][3],
    )
    angle = np.deg2rad(ellipse_params[init_frame][4])
    dx_major = major_axis / 2 * np.cos(angle)
    dy_major = major_axis / 2 * np.sin(angle)
    dx_minor = minor_axis / 2 * np.cos(angle + np.pi / 2)
    dy_minor = minor_axis / 2 * np.sin(angle + np.pi / 2)

    (line_major,) = ax[0].plot(
        [xc - dx_major, xc + dx_major],
        [yc - dy_major, yc + dy_major],
        color="blue",
        linestyle="--",
        linewidth=1,
    )
    (line_minor,) = ax[0].plot(
        [xc - dx_minor, xc + dx_minor],
        [yc - dy_minor, yc + dy_minor],
        color="green",
        linestyle="--",
        linewidth=1,
    )

    # Add text annotations for width and height in angstroms
    text_major = ax[0].text(
        xc + dx_major,
        yc + dy_major,
        f"Width: {major_axis * pixel_size:.2f} Å",
        color="blue",
        fontsize=8,
        ha="center",
    )
    text_minor = ax[0].text(
        xc + dx_minor,
        yc + dy_minor,
        f"Height: {minor_axis * pixel_size:.2f} Å",
        color="green",
        fontsize=8,
        ha="center",
    )

    ax[0].set(title=f"Frame {init_frame}")
    ax[1].set(title=f"Mask {areas[init_frame]:.2f} Å²")
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax_slider, "Frame", 0, lp_data.shape[0] - 1, valinit=init_frame, valstep=1
    )
    fig.colorbar(image, ax=ax)

    def update(val):
        frame = int(slider.val)  # Get the current slider value
        image.set(data=lp_data[frame], clim=clims[frame])
        mask.set(data=masks[frame], clim=(0, 1))
        ellipse.set_center((ellipse_params[frame][1], ellipse_params[frame][0]))
        ellipse.width = ellipse_params[frame][2]
        ellipse.height = ellipse_params[frame][3]
        ellipse.angle = ellipse_params[frame][4]

        # Update the width and height lines
        xc, yc = ellipse_params[frame][1], ellipse_params[frame][0]
        major_axis, minor_axis = ellipse_params[frame][2], ellipse_params[frame][3]
        angle = np.deg2rad(ellipse_params[frame][4])
        dx_major = major_axis / 2 * np.cos(angle)
        dy_major = major_axis / 2 * np.sin(angle)
        dx_minor = minor_axis / 2 * np.cos(angle + np.pi / 2)
        dy_minor = minor_axis / 2 * np.sin(angle + np.pi / 2)

        line_major.set_data(
            [xc - dx_major, xc + dx_major],
            [yc - dy_major, yc + dy_major],
        )
        line_minor.set_data(
            [xc - dx_minor, xc + dx_minor],
            [yc - dy_minor, yc + dy_minor],
        )

        # Update text annotations for width and height in angstroms
        text_major.set_position((xc + dx_major, yc + dy_major))
        text_major.set_text(f"Width: {major_axis * pixel_size:.2f} Å")
        text_minor.set_position((xc + dx_minor, yc + dy_minor))
        text_minor.set_text(f"Height: {minor_axis * pixel_size:.2f} Å")

        ax[0].set_title(f"Frame {frame}")
        ax[1].set(title=f"Mask {areas[frame]:.2f} Å²")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    plt.close()

    np.save("results/filtered_particles.npy", lp_data)
    np.save("results/masks.npy", masks)

    # Save area data and ellipse parameters to a CSV file
    with open("results/dimensions_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Frame", "Area", "Yc", "Xc", "Minor Axis", "Major Axis", "Angle"]
        )
        for i, (area, params) in enumerate(zip(areas, ellipse_params)):
            writer.writerow([i, area, *params])

    plt.style.use(matplotx.styles.pitaya_smoothie["light"])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(areas, bins=50)
    ax.set(xlim=(000, 10000), ylabel="Counts", xlabel="Area [Å^2]")
    plt.show()
    plt.close()

    init_frame = 60
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    image = ax[0].imshow(lp_data[init_frame], cmap="gray_r", clim=clims[init_frame])
    mask = ax[1].imshow(masks[init_frame], cmap="gray_r")
    ellipse = Ellipse(
        (ellipse_params[init_frame][1], ellipse_params[init_frame][0]),
        width=ellipse_params[init_frame][2],
        height=ellipse_params[init_frame][3],
        angle=ellipse_params[init_frame][4],
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    ax[0].add_patch(ellipse)

    # Plot the width and height lines
    xc, yc = ellipse_params[init_frame][1], ellipse_params[init_frame][0]
    major_axis, minor_axis = (
        ellipse_params[init_frame][2],
        ellipse_params[init_frame][3],
    )
    angle = np.deg2rad(ellipse_params[init_frame][4])
    dx_major = major_axis / 2 * np.cos(angle)
    dy_major = major_axis / 2 * np.sin(angle)
    dx_minor = minor_axis / 2 * np.cos(angle + np.pi / 2)
    dy_minor = minor_axis / 2 * np.sin(angle + np.pi / 2)

    (line_major,) = ax[0].plot(
        [xc - dx_major, xc + dx_major],
        [yc - dy_major, yc + dy_major],
        color="magenta",
        linestyle="--",
        linewidth=1,
    )
    (line_minor,) = ax[0].plot(
        [xc - dx_minor, xc + dx_minor],
        [yc - dy_minor, yc + dy_minor],
        color="lime",
        linestyle="--",
        linewidth=1,
    )

    # Add text annotations for width and height in angstroms
    text_major = ax[0].text(
        3,
        25,
        f"Width: {major_axis * pixel_size:.2f} Å",
        color="magenta",
        fontsize=14,
        ha="left",
        path_effects=[pe.withStroke(linewidth=2, foreground="k")],
        fontfamily="Century Gothic",
    )
    text_minor = ax[0].text(
        3,
        10,
        f"Height: {minor_axis * pixel_size:.2f} Å",
        color="lime",
        fontsize=14,
        ha="left",
        path_effects=[pe.withStroke(linewidth=2, foreground="k")],
        fontfamily="Century Gothic",
    )

    ax[0].set(title=f"Frame {init_frame}")
    ax[1].set(title=f"Mask {areas[init_frame]:.2f} Å²")
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    fig.savefig("results/ellipse.svg", transparent=True, bbox_inches="tight")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main_mask()
