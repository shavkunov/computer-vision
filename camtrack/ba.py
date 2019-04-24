from collections import namedtuple
from typing import List, Tuple

import numpy as np
from scipy.optimize import approx_fprime
from scipy.sparse import csr_matrix

from _camtrack import *
from corners import FrameCorners

ProjectionError = namedtuple('ProjectionError', ('frame_index', 'index_3d', 'index_2d'))


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          id2pos: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    print("running bundle adjustment")

    prev_id2pos = id2pos.copy()
    for idx, item in enumerate(id2pos):
        if item is None:
            id2pos[idx] = np.zeros(3)
    id2pos = np.array(id2pos)

    proj_mats = [np.dot(intrinsic_mat, view_mat) for view_mat in view_mats]
    projection_errors = []
    point3d_indices = set()

    for idx, (proj_mat, corners) in enumerate(zip(proj_mats, list_of_corners)):
        indices = []
        for id in corners.ids:
            if id2pos[id] is not None:
                indices.append(id)

        indices = np.array(indices, dtype=np.int32)

        indices_2d = []
        for num, id in enumerate(corners.ids):
            if id in indices:
                indices_2d.append(num)
        indices_2d = np.array(indices_2d, dtype=np.int32)
        indices_3d = indices[:, 0]
        inlier_indices = calc_inlier_indices(id2pos[indices_3d], corners.points[indices_2d],
                                             proj_mat, max_reprojection_error)

        for id in inlier_indices:
            id_3d = indices_3d[id]
            id_2d = indices_2d[id]
            point3d_indices.add(id_3d)
            projection_errors.append(ProjectionError(frame_index=idx, index_3d=id_3d, index_2d=id_2d))

    point3d_indices = list(sorted(point3d_indices))
    idx2pos = {}
    mx_params = 6 * len(view_mats)

    rodrigues_translation = np.zeros(6 * len(view_mats))
    for idx, mat in enumerate(view_mats):
        r, t = view_mat3x4_to_rodrigues_and_translation(mat)
        rodrigues_translation[6 * idx: 6 * idx + 3] = r[:, 0]
        rodrigues_translation[6 * idx + 3: 6 * idx + 6] = t[:]

    p = np.concatenate([rodrigues_translation, id2pos[point3d_indices].reshape(-1)])
    for idx, point_ind in enumerate(point3d_indices):
        idx2pos[point_ind] = mx_params + 3 * idx

    p = optimize_params(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat)

    for idx, _ in enumerate(view_mats):
        r_vec = p[6 * idx: 6 * idx + 3].reshape(3, 1)
        t_vec = p[6 * idx + 3: 6 * idx + 6].reshape(3, 1)

        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        view_mats[idx] = view_mat

    for idx, id in enumerate(point3d_indices):
        prev_id2pos[id] = p[mx_params + 3 * idx: mx_params + 3 * idx + 3]

    return view_mats, prev_id2pos


def reprojection_error(vec, point2d, intrinsic_mat):
    point3d = vec[6:9]

    r_vec = vec[0:3].reshape(3, 1)
    t_vec = vec[3:6].reshape(3, 1)
    view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)

    proj_mat = np.dot(intrinsic_mat, view_mat)
    point3d_hom = np.hstack((point3d, 1))
    proj_point2d = np.dot(proj_mat, point3d_hom)
    proj_point2d = proj_point2d / proj_point2d[2]
    proj_point2d = proj_point2d.T[:2]

    return np.linalg.norm((point2d - proj_point2d).reshape(-1))


def compute_jacobian(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat):
    print("computing jacobian")

    rows = []
    columns = []
    values = []

    for row, pe in enumerate(projection_errors):
        vec = np.zeros(9)
        vec[:6] = p[6 * pe.frame_index: 6 * pe.frame_index + 6]
        vec[6:] = p[idx2pos[pe.index_3d]: idx2pos[pe.index_3d] + 3]

        point2d = list_of_corners[pe.frame_index].points[pe.index_2d]
        partial_derivatives = approx_fprime(vec, lambda v: reprojection_error(v, point2d, intrinsic_mat),
                                            np.full_like(vec, 1e-9))

        for i in range(6):
            rows.append(row)
            columns.append(6 * pe.frame_index + i)
            values.append(partial_derivatives[i])

        for i in range(3):
            rows.append(row)
            columns.append(idx2pos[pe.index_3d] + i)
            values.append(partial_derivatives[6 + i])

    return csr_matrix((values, (rows, columns)), shape=(len(projection_errors), len(p)))


def reprojection_errors(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat):
    errors = np.zeros(len(projection_errors))

    for idx, pe in enumerate(projection_errors):
        vec = np.zeros(9)
        vec[:6] = p[6 * pe.frame_index: 6 * pe.frame_index + 6]

        point_pos = idx2pos[pe.index_3d]
        vec[6:] = p[point_pos: point_pos + 3]

        point2d = list_of_corners[pe.frame_index].points[pe.index_2d]
        errors[idx] = reprojection_error(vec, point2d, intrinsic_mat)

    return errors


def optimize_params(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat):
    n = 3 * len(list_of_corners)

    start_error = reprojection_errors(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat).sum()
    print(f'start error: {start_error}')

    lmbda = 2.0
    for _ in range(10):
        J = compute_jacobian(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat)
        JTJ = J.T.dot(J).toarray()
        JTJ += lmbda * np.diag(np.diag(JTJ))

        U = JTJ[:n, :n]
        W = JTJ[:n, n:]

        V = JTJ[n:, n:]
        V = np.linalg.inv(V)

        g = J.toarray().T.dot(reprojection_errors(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat))
        A = U - W.dot(V).dot(W.T)
        B = W.dot(V).dot(g[n:]) - g[:n]

        try:
            delta_c = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            lmbda *= 5.0
            continue

        delta_x = V.dot(-g[n:] - W.T.dot(delta_c))
        new_p = p.copy() + np.hstack((delta_c, delta_x))

        error = reprojection_errors(projection_errors, list_of_corners, idx2pos, p, intrinsic_mat).sum()
        if error > start_error:
            lmbda *= 5.0
            print(f"new lambda : {lmbda}")
        else:
            p = new_p
            start_error = error
            lmbda /= 5.0
            print(f"new lambda : {lmbda}")

    return p
