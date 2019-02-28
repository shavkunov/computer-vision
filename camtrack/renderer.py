#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL import GLUT
from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import data3d

opengl_matrix = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


def build_color_program():
    color_vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
        in vec3 color_in;
        out vec3 color;

        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            color = color_in;
        }""",
        GL.GL_VERTEX_SHADER
    )
    points_color_fragment_shader = shaders.compileShader(
        """
        #version 130

        in vec3 color;
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )
    return shaders.compileProgram(
        color_vertex_shader, points_color_fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self.cam_track = tracked_cam_track
        self.cam_params = tracked_cam_parameters

        self.points_vbo = vbo.VBO(np.array(point_cloud.points, dtype=np.float32))
        self.colors_vbo = vbo.VBO(np.array(point_cloud.colors, dtype=np.float32))
        positions = []
        for p in tracked_cam_track:
            positions.append(p.t_vec)

        self.track_pos = vbo.VBO(np.array(positions, dtype=np.float32))
        self.track_col = vbo.VBO(np.array([[1, 1, 1]] * len(tracked_cam_track)).astype(np.float32))
        self.color_program = build_color_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    @staticmethod
    def build_transform(tr_vec, rot_matrix):
        top = np.hstack([rot_matrix, tr_vec.reshape(-1, 1)])
        return np.vstack([top, [0, 0, 0, 1]]).astype(np.float32)

    @staticmethod
    def build_view(tr_vec, rot_matrix):
        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[:, 3] = np.append(-tr_vec, 1)
        rot_inv = np.linalg.inv(rot_matrix)
        rot_inv = np.vstack((rot_inv, [0.0, 0.0, 0.0]))
        rot_inv = np.hstack((rot_inv, [[0.0], [0.0], [0.0], [1.0]]))

        return rot_inv.dot(view_matrix)

    @staticmethod
    def build_projection(fov_y, ratio, near, far):
        fy = 1 / np.tan(fov_y / 2)
        fx = fy / ratio
        t1 = -(far + near) / (far - near)
        t2 = -2 * far * near / (far - near)

        return np.array([
            [fx, 0, 0, 0],
            [0, fy, 0, 0],
            [0, 0, t1, t2],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
        p = self.build_projection(camera_fov_y, ratio, 0.01, 40)
        v = self.build_view(camera_tr_vec, camera_rot_mat)
        mvp = np.dot(p, np.dot(v, opengl_matrix))
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        pos = int(tracked_cam_track_pos_float)

        # cloud points
        buffers = [
            ('position', self.points_vbo),
            ('color_in', self.colors_vbo)
        ]
        self.render(mvp, buffers, GL.GL_POINTS,
                    self.points_vbo.size // 3)

        # camera track
        buffers = [
            ('position', self.track_pos),
            ('color_in', self.track_col)
        ]
        self.render(mvp, buffers, GL.GL_LINE_STRIP, len(self.cam_track))

        # frustrum
        frustrum_points = self.get_frustrum_points(
            self.cam_track[pos],
            self.cam_params.fov_y,
            self.cam_params.aspect_ratio)

        borders = np.array([
            frustrum_points[0], frustrum_points[1],
            frustrum_points[1], frustrum_points[2],
            frustrum_points[2], frustrum_points[3],
            frustrum_points[3], frustrum_points[0],
            self.cam_track[pos].t_vec, frustrum_points[0],
            self.cam_track[pos].t_vec, frustrum_points[1],
            self.cam_track[pos].t_vec, frustrum_points[2],
            self.cam_track[pos].t_vec, frustrum_points[3]], dtype=np.float32)

        points_vbo = vbo.VBO(borders.reshape(-1))
        colors_vbo = vbo.VBO(np.array([1, 1, 0] * len(borders), dtype=np.float32))
        buffers = [
            ('position', points_vbo),
            ('color_in', colors_vbo)
        ]
        self.render(mvp, buffers, GL.GL_LINES, len(borders))

        GLUT.glutSwapBuffers()

    @staticmethod
    def get_frustrum_points(pos, fov_y, ratio):
        z = 20.0
        y = z * np.tan(fov_y / 2)
        x = y * ratio

        c1 = pos.r_mat.dot(np.array([x, y, z], dtype=np.float32)) + pos.t_vec
        c2 = pos.r_mat.dot(np.array([x, -y, z], dtype=np.float32)) + pos.t_vec
        c3 = pos.r_mat.dot(np.array([-x, -y, z], dtype=np.float32)) + pos.t_vec
        c4 = pos.r_mat.dot(np.array([-x, y, z], dtype=np.float32)) + pos.t_vec

        return np.array([c1, c2, c3, c4], dtype=np.float32)

    def render(self, mvp, buffers, draw_mode, draw_cnt):
        shaders.glUseProgram(self.color_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.color_program, 'mvp'), 1, True, mvp)

        for attr, buffer in buffers:
            buffer.bind()
            loc = GL.glGetAttribLocation(self.color_program, attr)
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, False, 0, buffer)

        GL.glDrawArrays(draw_mode, 0, draw_cnt)

        for attr, buffer in reversed(buffers):
            loc = GL.glGetAttribLocation(self.color_program, attr)
            GL.glDisableVertexAttribArray(loc)
            buffer.unbind()

        shaders.glUseProgram(0)
