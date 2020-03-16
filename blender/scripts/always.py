#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

########################################################################
# This file is part of ChArUCo.
#
# ChArUCo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ChArUCo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
########################################################################

'''
Lancé à chaque frame durant tout le jeu.
'''


from bge import logic as gl
import mathutils
import ast


def main():
    try:
        raw_data = gl.multi.receive()
    except:
        raw_data = None

    if raw_data:
        data_to_var(raw_data)
        # #rot_filter()
        set_rot()
        set_trans()


def rot_filter():
    for i in range(3):
        if abs(gl.rvec[i] - gl.rvec_old[i]) > 0.2:
            gl.rvec[i] = gl.rvec_old[i]

    gl.rvec_old = gl.rvec

def set_rot():
    """ -0.32 0.01 0.48"""

    alpha = -(gl.rvec[0] + 2.6)
    beta = gl.rvec[1]
    gamma = gl.rvec[2] - 0.2

    print("angle", round(alpha, 2), round(beta, 2), round(gamma, 2))

    # set objects orientation with alpha, beta, gamma in radians
    rot_en_euler = mathutils.Euler([alpha, beta-0.2, gamma+0.2])

    # apply to objects local orientation if ok
    gl.cube.localOrientation = rot_en_euler.to_matrix()


def set_trans():

    x = -gl.tvec[0] * 50
    y = gl.tvec[2] * 50 - 4
    z = -gl.tvec[1]*50 - 2
    print("position", round(x, 2), round(y, 2), round(z, 2))
    gl.cube.position = x, y, z


def datagram_to_dict(data):
    """Décode le message. Retourne un dict ou None."""

    try:
        dec = data.decode("utf-8")
    except:
        print("Décodage UTF-8 impossible")
        dec = data

    try:
        msg = ast.literal_eval(dec)
    except:
        print("ast.literal_eval impossible")
        print("Ajouter ast dans les import")
        msg = dec

    if isinstance(msg, dict):
        return msg
    else:
        print("Message reçu: None")
        return None


def data_to_var(data):
    data = datagram_to_dict(data)
    # #print(data)
    if data:
        if "rvec" in data:
            gl.rvec = data["rvec"][0][0]
    if data:
        if "tvec" in data:
            gl.tvec = data["tvec"][0][0]
