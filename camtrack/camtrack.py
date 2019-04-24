#! /usr/bin/env python3
from cv2.cv2 import findHomography, findEssentialMat, decomposeEssentialMat, solvePnPRansac

from _camtrack import remove_correspondences_with_ids

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

import cv2
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *
from ba import run_bundle_adjustment


def first_init(frame_corners1, frame_corners2, intrinsic_mat, tr_parameters):
    correspondences = build_correspondences(frame_corners1, frame_corners2)
    if correspondences.points_1.shape[0] < 5:
        return None, None

    points1 = correspondences.points_1
    points2 = correspondences.points_2

    E, e_mask = cv2.findEssentialMat(points1, points2,
                                     cameraMatrix=intrinsic_mat, method=cv2.RANSAC, threshold=1.0, prob=0.99999)
    corrs = remove_correspondences_with_ids(correspondences, np.where(e_mask == 0)[0])

    if E is None or E.shape != (3, 3):
        return None, None

    R1, R2, t_d = decomposeEssentialMat(E)

    size = 0
    pose = None
    for R, t in [(R1, t_d), (R1, -t_d), (R2, t_d), (R2, -t_d)]:
        view = pose_to_view_mat3x4(Pose(R.T, R.T @ -t))
        _, ids = triangulate_correspondences(corrs, eye3x4(), view, intrinsic_mat, tr_parameters)

        if ids.shape[0] > size:
            size = ids.shape[0]
            pose = Pose(R.T, R.T @ -t)

    if size == 0:
        return None, None

    return pose, size


def add2cloud(frame_id1, frame_id2, corner_storage, frame_matrix,
              intrinsic_mat, tr_parameters, id2pos):
    correspondences = build_correspondences(corner_storage[frame_id1], corner_storage[frame_id2])
    positions, ids = triangulate_correspondences(
        correspondences, frame_matrix[frame_id1], frame_matrix[frame_id2],
        intrinsic_mat, tr_parameters)

    for pos, id in zip(positions, ids):
        if id2pos[id] is None:
            id2pos[id] = pos


def frame_points(frame_corners, id2pos):
    mask = np.ones(frame_corners.ids.shape[0], dtype=np.bool)

    for i in range(mask.shape[0]):
        if id2pos[frame_corners.ids[i][0]] is None:
            mask[i] = 0

    return frame_corners.ids[mask, 0], frame_corners.points[mask]


def _track_camera(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    tr_parameters = \
        TriangulationParameters(max_reprojection_error=1.0, min_triangulation_angle_deg=3.0, min_depth=0.1)

    poses, qualities = zip(*[first_init(corner_storage[0], i, intrinsic_mat, tr_parameters) for i in corner_storage])
    idx = np.nanargmax(np.array(qualities, dtype=np.float32))

    frame_matrix = [None] * len(corner_storage)
    frame_matrix[0] = eye3x4()
    frame_matrix[idx] = pose_to_view_mat3x4(poses[idx])

    max_id = np.amax(np.concatenate([corners.ids for corners in corner_storage]))
    id2pos = [None] * (max_id + 1)
    add2cloud(0, idx, corner_storage, frame_matrix, intrinsic_mat, tr_parameters, id2pos)

    current_frames = set(range(len(corner_storage)))
    current_frames.remove(0)
    current_frames.remove(idx)
    step = 0
    while len(current_frames) > 0:
        mask = np.ones(len(frame_matrix))
        fail = False
        while True:
            num = 0
            best_frame_idx = -1

            for f_index, mx in enumerate(frame_matrix):
                if mx is not None or not mask[f_index]:
                    continue

                cur_num = len(frame_points(corner_storage[f_index], id2pos)[0])
                if cur_num > num:
                    num = cur_num
                    best_frame_idx = f_index

            point_ids, image_points = frame_points(corner_storage[best_frame_idx], id2pos)
            if point_ids.shape[0] < 6:
                fail = True
                break

            object_points = np.array([id2pos[point_id] for point_id in point_ids])
            retval, rvec, tvec, inliers = solvePnPRansac(object_points, image_points, intrinsic_mat, None)

            if not retval:
                mask[best_frame_idx] = False
                continue

            inliers = inliers.reshape(-1)
            inner_mask = np.ones(point_ids.shape[0], dtype=np.bool)
            inner_mask[inliers] = 0

            for outlier_id in point_ids[inner_mask]:
                id2pos[outlier_id] = None

            frame_matrix[best_frame_idx] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            print('Added frame {}'.format(best_frame_idx))
            step += 1
            if step % 5 == 0 and step > 7:
                frame_matrix[best_frame_idx - 5: best_frame_idx], id2pos = \
                    run_bundle_adjustment(intrinsic_mat,
                                          list(corner_storage)[best_frame_idx - 5: best_frame_idx],
                                          tr_parameters.max_reprojection_error,
                                          frame_matrix[best_frame_idx - 5: best_frame_idx],
                                          id2pos)

            break

        if fail:
            break

        for id, matrix in enumerate(frame_matrix):
            if id == best_frame_idx or matrix is None:
                continue

            add2cloud(id, best_frame_idx, corner_storage, frame_matrix,
                      intrinsic_mat, tr_parameters, id2pos)

        current_frames.remove(best_frame_idx)

    for fid in range(len(frame_matrix)):
        if frame_matrix[fid] is None:
            frame_matrix[fid] = eye3x4()

    indices = []
    for id, pos in enumerate(id2pos):
        if pos is not None:
            indices.append(id)

    return frame_matrix, PointCloudBuilder(np.array(indices), np.array([id2pos[d_idx] for d_idx in indices]))


def track_and_calc_colors(camera_parameters: CameraParameters, corner_storage: CornerStorage,
                          frame_sequence_path: str) -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
