#!/usr/bin/env python3

import argparse
import math
import sys

import numpy as np

from PIL import Image
from scipy.interpolate import splprep, splev


def get_perspective_transform_coeffs(p_src, p_dest):
    '''
    Get coefficients for a perspective transformation from `p_src` to `p_dest`.

    https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/
    '''
    matrix = []
    for (X, Y), (x, y) in zip(p_src, p_dest):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(p_src).reshape(8)

    result = np.linalg.solve(A, B)
    return np.array(result).reshape(8)


def get_perspective_transform_bounds(grid_size, top_midpt):
    '''
    Compute perspective transform boundaries. The midpoint of the top of the
    image will be moved to `top_midpt`. The top two corners will have the same
    Y coordinate as `top_midpt`. The top corner on the opposite half of the
    image that the midpoint is displacing towards will be displaced by 1.5
    times the amount. The bottom two corners will not be changed.

    Returns (top-left, top-right, bottom-right, bottom-left)
    '''

    # Midpoint displacement
    d = top_midpt[0] - grid_size[0] // 2
    # Displacment ratio of opposing corner
    ed_ratio = 1.5

    tl = (0, top_midpt[1])
    tr = (grid_size[0] - 1, top_midpt[1])

    if d < 0:
        tr = (tr[0] + ed_ratio * d, tr[1])
    elif d > 0:
        tl = (tl[0] + ed_ratio * d, tl[1])

    return (
        tl,
        tr,
        (grid_size[0] - 1, grid_size[1] - 1),
        (0, grid_size[1] - 1),
    )


def get_perspective_transform(grid_size, top_midpt):
    '''
    Get perspective transform coefficients for parrot head movement.
    '''

    src = (
        (0, 0),
        (grid_size[0] - 1, 0),
        (grid_size[0] - 1, grid_size[1] - 1),
        (0, grid_size[1] - 1),
    )
    dest = get_perspective_transform_bounds(grid_size, top_midpt)

    return get_perspective_transform_coeffs(src, dest)


def pick_equally_spaced(pts, n, endpoint=False):
    '''
    Compute equally spaced points from a list of points using linear
    approximation. If `endpoint` is true, then include the endpoint in the
    results.
    '''

    assert len(pts) > 1

    # Compute total length
    total_length = 0.0
    lengths = []

    for i in range(1, len(pts)):
        prev = pts[i - 1]
        cur = pts[i]

        length = math.sqrt((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2)
        total_length += length
        lengths.append(length)

    segments = n - 1 if endpoint else n
    segment_length = total_length / segments

    result = np.zeros((n, 2), dtype=np.float)
    result_i = 0
    length_i = 0

    cur_length = 0.0
    target_length = 0.0

    while result_i < segments:
        length = lengths[length_i]

        if cur_length <= target_length < cur_length + length:
            prev = pts[length_i]
            cur = pts[length_i + 1]

            t = (target_length - cur_length) / length
            result[result_i] = (
                (1 - t) * prev[0] + t * cur[0],
                (1 - t) * prev[1] + t * cur[1],
            )
            result_i += 1

            target_length += segment_length
        else:
            cur_length += length
            length_i += 1

    if endpoint:
        result[result_i] = pts[-1]

    return result


def get_parrot_animation_points(grid_size, n):
    '''
    Get `n` equally spaced points along the animation path of the party parrot.
    Each point represents the top of the bird's head.
    '''

    # Points on a 35x35 grid
    pts = np.array((
        (15.0, 5.0),
        (19.0, 6.0),
        (25.0, 5.0),
        (24.0, 2.0),
        (17.0, 1.0),
        (12.0, 1.0),
        (10.0, 4.0),
        (10.0, 7.0),
        (12.0, 6.0),
        (14.0, 4.0),
    ))

    # Scale to specified resolution
    for axis, size in enumerate(grid_size):
        pts[..., axis] = pts[..., axis] / 35.0 * size

    # Interpolate curve to fit animation path
    tck, u = splprep(pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    # Get equally spaced (euclidean) points along the curve
    interp_pts = np.column_stack((x_new, y_new))
    interp_pts_norm = pick_equally_spaced(interp_pts, n)

    return interp_pts_norm


def parrotize(im, hue, coeffs):
    '''
    Parrotize the specified image by change the hue to `hue` and doing a
    perspective transform with coefficients `coeffs`.
    '''

    assert 0 <= hue < 360

    alpha_channel = im.split()[-1] if im.mode == 'RGBA' else None
    hsv_im = im.convert('HSV')
    hsv_px = np.array(hsv_im)

    # Colorize
    hsv_px[:, :, 0] = round(hue / 360 * 256)

    new_im = Image.fromarray(hsv_px, 'HSV').convert('RGB')
    if alpha_channel:
        new_im.putalpha(alpha_channel)

        rgba_px = np.array(new_im)
        clamp_val = (35, 35, 35, 255)

        # Clamp minimum opaque RGB value to 35 to prevent RGB -> indexed
        # conversion from treating dark colors as transparent
        rgba_px[((rgba_px < clamp_val) ==
                (True, True, True, False)).all(axis=2)] = clamp_val

        # Convert to indexed color here because, for some reason, having the
        # GIF writer do it results in weird artifacts
        new_im = Image.fromarray(rgba_px, 'RGBA').convert('RGB').convert('P')

    return new_im.transform(im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames in output file')
    parser.add_argument('--time', type=int, default=1000,
                        help='Duration of animation in milliseconds')
    parser.add_argument('source_file', help='Source image')
    parser.add_argument('target_file', help='Target image')

    args = parser.parse_args()

    frame_duration = round(args.time / args.frames)
    if frame_duration < 10:
        parser.error('Frame duration (%dms) must be greater than 10ms'
                     % frame_duration)

    print('Frame duration: %dms\n' % (frame_duration // 10 * 10))

    im = Image.open(args.source_file)

    # Compute rainbox colors
    hues = np.linspace(0, 360, num=args.frames, endpoint=False)

    # Compute perspective transform coefficients
    frame_coeffs = []
    for pt in get_parrot_animation_points(im.size, args.frames):
        frame_coeffs.append(get_perspective_transform(im.size, pt))

    frames = []

    for hue, coeffs in zip(hues, frame_coeffs):
        print('Processing frame #%d' % len(frames))
        frames.append(parrotize(im, hue, coeffs))

    frames[0].save(args.target_file, save_all=True, append_images=frames[1:],
                   duration=frame_duration, loop=0, transparency=0, disposal=2)


if __name__ == '__main__':
    main()
