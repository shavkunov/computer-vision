#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli
from sklearn.neighbors import KDTree
# based on : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

def check_corner(corner_pos, corner_id, tree, min_dist):
    dist, t_ids = tree.query([corner_pos], k=3)

    check = True
    for pos_id, d in zip(t_ids[0], dist[0]):
        if pos_id != corner_id and d < min_dist / 2:
            check = False

    return check


def filter_points(positions, ids, tree, min_dist):
    new_ids = []
    ret_pos = []

    for id in range(len(positions)):
        corner_pos = positions[id]
        corner_id = id

        if check_corner(corner_pos, corner_id, tree, min_dist):
            new_ids.append(ids[id])
            ret_pos.append(corner_pos)

    return np.array(ret_pos), np.array(new_ids)

def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    min_distance = 5
    corners_amount = 500
    feature_params = dict(maxCorners=corners_amount,
                          qualityLevel=0.3,
                          minDistance=min_distance,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params)

    ids = np.array(range(len(p0)))
    corners = FrameCorners(ids, p0.reshape((-1, 2)), np.full(len(p0), 10))

    builder.set_corners_at_frame(0, corners)
    image0_byte = np.uint8(image_0 * 255.0)
    next_id = len(ids)

    positions = []
    for pos in p0:
        positions.append(pos[0])

    tree = KDTree(positions, leaf_size=10, metric="l2")
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image1_byte = np.uint8(image_1 * 255.0)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image0_byte, image1_byte, p0, None, **lk_params)

        new_ids = np.array([id for i, id in enumerate(ids) if st[i][0] == 1])
        p0 = np.array([p for i, p in enumerate(p1) if st[i][0] == 1])

        new_positions = []
        for i, id in enumerate(ids):
            if st[i][0] == 1:
                new_positions.append(positions[i])

        positions = new_positions
        ids = new_ids
        tree = KDTree(positions, leaf_size=10, metric="l2")

        # filter points
        positions, ids = filter_points(positions, ids, tree, min_distance)

        # add new ones
        if len(p0) < corners_amount:
            p2 = cv2.goodFeaturesToTrack(image_1, mask=None, **feature_params)
            if p2 is not None:
                new_positions = []
                new_ids = []
                p_new = []

                for pos in p2:
                    if len(ids) + len(new_ids) == corners_amount:
                        break

                    x, y = pos[0]
                    c = True
                    for i in range(len(positions)):
                        x1, y1 = positions[i]

                        if np.linalg.norm(np.array([x - x1, y - y1])) < min_distance:
                            c = False
                            break

                    if not c:
                        continue

                    new_positions.append(pos[0])
                    new_ids.append(next_id)
                    p_new.append(pos)
                    next_id += 1

                if len(p_new) > 0:
                    ids = np.concatenate([ids, new_ids])
                    p0 = np.concatenate([p0, p_new])
                    positions = np.concatenate([positions, new_positions])

        corners = FrameCorners(ids, p0.reshape((-1, 2)), np.full(len(p0), 10))
        #print(len(ids))
        image0_byte = image1_byte
        builder.set_corners_at_frame(frame, corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
