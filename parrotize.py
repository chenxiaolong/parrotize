#!/usr/bin/env python3

import argparse
import itertools
import math
import sys

import numpy as np

from scipy.interpolate import splprep, splev
from wand.image import Image


def get_perspective_transform(grid_size, top_midpt):
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

    # Source coordinate, followed by destination coordinate
    return (
        # Top left
        0, 0,
        tl[0], tl[1],
        # Top right
        grid_size[0] - 1, 0,
        tr[0], tr[1],
        # Bottom right
        grid_size[0] - 1, grid_size[1] - 1,
        grid_size[0] - 1, grid_size[1] - 1,
        # Bottom left
        0, grid_size[1] - 1,
        0, grid_size[1] - 1,
    )


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


def parrotize(im, hue, ptransform):
    '''
    Parrotize the specified image by change the hue to `hue` and doing a
    perspective transform with the quadrilateral mapping in `ptransform`.
    The `im` image will be modified in place.
    '''

    assert 0 <= hue < 360

    # Convert to HSV and keep it in that color space. ImageMagick preserves the
    # alpha channel and will use it when saving the image.
    im.transform_colorspace('hsv')

    # 'red' == hue channel
    hue_value = hue / 360 * im.quantum_range
    im.evaluate(operator='set', value=hue_value, channel='red')

    im.distort('perspective', ptransform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=25,
                        help='Number of frames in output file')
    parser.add_argument('--time', type=int, default=500,
                        help='Duration of animation in milliseconds')
    parser.add_argument('source_file', help='Source image')
    parser.add_argument('target_file', help='Target image')

    args = parser.parse_args()

    frame_duration = round(args.time / args.frames)
    if frame_duration < 10:
        parser.error('Frame duration (%dms) must be greater than 10ms'
                     % frame_duration)

    print('Frame duration: %dms\n' % (frame_duration // 10 * 10))

    # Compute rainbow colors
    hues = np.linspace(0, 360, num=args.frames, endpoint=False)

    with Image(filename=args.source_file) as im:
        print(im)
        print(im.size)

        # Compute perspective transform mapping
        ptransforms = []
        for pt in get_parrot_animation_points(im.size, args.frames):
            ptransforms.append(get_perspective_transform(im.size, pt))

        with Image() as im_out:
            for i, hue, ptransform in zip(itertools.count(), hues, ptransforms):
                print('Processing frame #%d' % i)
                with im.clone() as im_frame:
                    im_frame.dispose = 'background'
                    im_frame.virtual_pixel = 'transparent'

                    parrotize(im_frame, hue, ptransform)

                    im_out.sequence.append(im_frame)
                    im_out.sequence[i].delay = frame_duration // 10

            im_out.type = 'optimize'
            im_out.save(filename=args.target_file)


if __name__ == '__main__':
    main()
