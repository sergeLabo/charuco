#!/usr/bin/env python
# coding: utf-8

"""

Extrait de la doc
https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/Projet+calibration-Paul.html

"""

import os
import numpy as np
import cv2
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from time import time, sleep
import json

from webcam import apply_all_cam_settings, apply_cam_setting
from myconfig import MyConfig
from multicast import Multicast

folder = "/media/data/3D/projets/charuco"


class WebcamSettings:

    def __init__(self, cam, cf):

        self.cam = cam
        self.cap = cv2.VideoCapture(self.cam)

        # L'objet config de aruco.ini
        self.cf = cf

        self.brightness = self.cf.conf['HD5000']['brightness']
        self.contrast = self.cf.conf['HD5000']['contrast']
        self.saturation = self.cf.conf['HD5000']['saturation']

        self.w_bal_temp_aut = self.cf.conf['HD5000']['w_bal_temp_aut']
        self.power_line_freq = self.cf.conf['HD5000']['power_line_freq']

        self.exposure_auto = self.cf.conf['HD5000']['exposure_auto']
        self.exposure_absolute = self.cf.conf['HD5000']['exposure_absolute']

        self.white_bal_temp = self.cf.conf['HD5000']['white_bal_temp']
        self.backlight_compensation = self.cf.conf['HD5000']['backlight_compensation']

        self.sharpness = self.cf.conf['HD5000']['sharpness']
        self.pan = self.cf.conf['HD5000']['pan']
        self.tilt = self.cf.conf['HD5000']['tilt']
        self.focus_absolute = self.cf.conf['HD5000']['focus_absolute']
        self.focus_auto = self.cf.conf['HD5000']['focus_auto']
        self.zoom_absolute = self.cf.conf['HD5000']['zoom_absolute']

        # Trackbars
        self.create_trackbar()
        self.set_init_tackbar_position()

    def create_trackbar(self):
        """
        brightness (int): min=30 max=255 step=1 default=133 value=50
        contrast (int): min=0 max=10 step=1 default=5 value=5
        saturation (int): min=0 max=200 step=1 default=83 value=100
        white_balance_temperature_auto (bool): default=1 value=0
        power_line_frequency (menu) : min=0 max=2 default=2 value=0
        white_balance_temperature (int): min=2800 max=10000 step=1 default=4500 value=10000
        backlight_compensation (int): min=0 max=10 step=1 default=0 value=1
        exposure_auto (menu): min=0 max=3 default=1 value=1
        exposure_absolute (int): min=5 max=20000 step=1 default=156 value=150

        sharpness int): min=0 max=50 step=1 default=25 value=25
        pan_absolute (int): min=-201600 max=201600 step=3600 default=0 value=0
        tilt (int): min=-201600 max=201600 step=3600 default=0 value=0
        focus_absolute (int): min=0 max=40 step=1 default=0 value=0
        focus_auto (bool): default=0 value=0
        zoom_absolute (int): min=0 max=10 step=1 default=0 value=0
        """

        cv2.namedWindow('Reglage')
        self.reglage_img = np.zeros((10, 1400, 3), np.uint8)

        cv2.createTrackbar('brightness', 'Reglage', 0, 255, self.onChange_brightness)
        cv2.createTrackbar('contrast', 'Reglage', 0, 10, self.onChange_contrast)
        cv2.createTrackbar('saturation', 'Reglage', 0, 200, self.onChange_saturation)
        cv2.createTrackbar('w_bal_temp_aut', 'Reglage', 0, 1, self.onChange_w_bal_temp_aut)
        cv2.createTrackbar('power_line_freq', 'Reglage', 0, 2, self.onChange_power_line_freq)
        cv2.createTrackbar('white_bal_temp', 'Reglage', 2800, 10000, self.onChange_white_bal_temp)
        cv2.createTrackbar('backlight_compensation', 'Reglage', 0, 10, self.onChange_backlight_compensation)
        cv2.createTrackbar('exposure_auto', 'Reglage', 0, 3, self.onChange_exposure_auto)
        cv2.createTrackbar('exposure_absolute', 'Reglage', 5, 20000, self.onChange_exposure_absolute)

        cv2.createTrackbar('sharpness', 'Reglage', 0, 50, self.onChange_sharpness)
        cv2.createTrackbar('pan', 'Reglage', -201600, 201600, self.onChange_pan)
        cv2.createTrackbar('tilt', 'Reglage', -201600, 201600, self.onChange_tilt)
        cv2.createTrackbar('focus_absolute', 'Reglage', 0, 40, self.onChange_focus_absolute)
        cv2.createTrackbar('focus_auto', 'Reglage', 0, 1, self.onChange_focus_auto)
        cv2.createTrackbar('zoom_absolute', 'Reglage', 0, 10, self.onChange_zoom_absolute)

    def set_init_tackbar_position(self):
        """setTrackbarPos(trackbarname, winname, pos) -> None"""

        cv2.setTrackbarPos('brightness', 'Reglage', self.brightness)
        cv2.setTrackbarPos('saturation', 'Reglage', self.saturation)
        cv2.setTrackbarPos('exposure_auto', 'Reglage', self.exposure_auto)
        cv2.setTrackbarPos('exposure_absolute', 'Reglage', self.exposure_absolute)
        cv2.setTrackbarPos('contrast', 'Reglage', self.contrast)
        cv2.setTrackbarPos('w_bal_temp_aut', 'Reglage', self.w_bal_temp_aut)
        cv2.setTrackbarPos('power_line_freq', 'Reglage', self.power_line_freq)
        cv2.setTrackbarPos('white_bal_temp', 'Reglage', self.white_bal_temp)
        cv2.setTrackbarPos('backlight_compensation', 'Reglage', self.backlight_compensation)

        cv2.setTrackbarPos('sharpness', 'Reglage', self.sharpness)
        cv2.setTrackbarPos('pan', 'Reglage', self.pan)
        cv2.setTrackbarPos('tilt', 'Reglage', self.tilt)
        cv2.setTrackbarPos('focus_absolute', 'Reglage', self.focus_absolute)
        cv2.setTrackbarPos('focus_auto', 'Reglage', self.focus_auto)
        cv2.setTrackbarPos('zoom_absolute', 'Reglage', self.zoom_absolute)

    def onChange_brightness(self, brightness):
        """min=30 max=255 step=1 default=133
        """
        if brightness < 30: brightness = 30
        self.brightness = brightness
        self.save_change('HD5000', 'brightness', brightness)

    def onChange_saturation(self, saturation):
        """min=0 max=200 step=1 default=83
        """
        self.saturation = saturation
        self.save_change('HD5000', 'saturation', saturation)

    def onChange_exposure_auto(self, exposure_auto):
        """min=0 max=3 default=1
        """
        self.exposure_auto = exposure_auto
        self.save_change('HD5000', 'exposure_auto', exposure_auto)

    def onChange_exposure_absolute(self, exposure_absolute):
        """min=5 max=20000 step=1 default=156
        """
        self.exposure_absolute = exposure_absolute
        self.save_change('HD5000', 'exposure_absolute', exposure_absolute)

    def onChange_contrast(self, contrast):
        """min=0 max=10 step=1
        """
        self.contrast =contrast
        self.save_change('HD5000', 'contrast', contrast)

    def onChange_w_bal_temp_aut(self, w_bal_temp_aut):
        """min=0 max=1
        """
        self.w_bal_temp_aut = w_bal_temp_aut
        self.save_change('HD5000', 'w_bal_temp_aut', w_bal_temp_aut)

    def onChange_power_line_freq(self, power_line_freq):
        """min=0 max=2
        """
        self.power_line_freq = power_line_freq
        self.save_change('HD5000', 'power_line_freq', power_line_freq)

    def onChange_white_bal_temp(self, white_bal_temp):
        """white_bal_temp    min=2800 max=10000
        """
        if white_bal_temp < 2800: white_bal_temp = 2800
        self.white_bal_temp = white_bal_temp
        self.save_change('HD5000', 'white_bal_temp', white_bal_temp)

    def onChange_backlight_compensation(self, backlight_compensation):
        """min=0 max=10 step=1
        """
        self.backlight_compensation = backlight_compensation
        self.save_change('HD5000', 'backlight_compensation', backlight_compensation)

    def onChange_sharpness(self, sharpness):
        """sharpness int): min=0 max=50 step=1 default=25 value=25"""

        self.sharpness = sharpness
        self.save_change('HD5000', 'sharpness', sharpness)

    def onChange_pan(self, pan_absolute):
        """min=-201600 max=201600 step=3600 default=0 value=0"""

        self.pan = pan
        self.save_change('HD5000', 'pan', pan)

    def onChange_tilt(self, tilt):
        """min=-201600 max=201600 step=3600 default=0 value=0"""

        self.tilt = tilt
        self.save_change('HD5000', 'tilt', tilt)

    def onChange_focus_absolute(self, focus_absolute):
        """min=0 max=40 step=1 default=0 value=0"""

        self.focus_absolute = focus_absolute
        self.save_change('HD5000', 'focus_absolute', focus_absolute)

    def onChange_focus_auto(self, focus_auto):
        """default=0 value=0"""

        self.focus_auto = focus_auto
        self.save_change('HD5000', 'focus_auto', focus_auto)

    def onChange_zoom_absolute(self, zoom_absolute):
        """min=0 max=10 step=1 default=0 value=0"""

        self.zoom_absolute = zoom_absolute
        self.save_change('HD5000', 'zoom_absolute', zoom_absolute)

    def save_change(self, section, key, value):

        self.cf.save_config(section, key, value)
        if section == 'HD5000':
            apply_cam_setting(self.cam, key, value)


class CharucoTest(WebcamSettings):

    def __init__(self, cam, cf):
        super().__init__(cam, cf)

        # Pour le multicast
        self.sender = None

        self.width = 1280
        self.height = 720
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        self.t = time()

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.board = aruco.CharucoBoard_create(3, 3, 1, 0.8, self.aruco_dict)

        self.folder = folder

    def create_markers_pdf(self):
        """Voir à quoi ça sert"""

        fig = plt.figure()
        nx = 8
        ny = 6
        for i in range(1, nx*ny+1):
            ax = fig.add_subplot(ny,nx, i)
            img = aruco.drawMarker(self.aruco_dict, i-1, 700)
            plt.imshow(img, cmap = mpl.cm.gray, interpolation="nearest")
            ax.axis("off")
        plt.show()
        plt.savefig(self.folder + "/markers.pdf")
        print("markers.pdf créé !")

    def create_chessboard_png(self):

        imboard = self.board.draw((1200, 1200))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(imboard, cmap = mpl.cm.gray, interpolation="nearest")
        ax.axis("off")
        cv2.imwrite(self.folder + "/chessboard.png", imboard)
        plt.grid()
        plt.show()
        print("Imprimer le damier de calibration!")

    def create_images_calibration(self):

        cap = cv2.VideoCapture(0)
        frameRate = cap.get(5) #frame rate
        a = 0
        print("Création de 150 images pour la calibration")

        sleep(5)
        a = 0
        self.t = time()
        while a < 150:
            name = self.folder + "/images/image_" +  str(int(a)) + ".jpg"
            ret, img = self.cap.read()
            # #im = cv2.resize(img, (2304, 1296),
                            # #interpolation=cv2.INTER_AREA)
            cv2.imshow("Original", img)

            # Affichage des trackbars
            cv2.imshow('Reglage', self.reglage_img)

            if time() - self.t > 1:
                cv2.imwrite(name, img)
                print([a]*10 )
                a += 1
                self.t = time()

            k = cv2.waitKey(10) & 0xFF
            if k == 27:  # ord('q'):
                break
        cv2.destroyAllWindows()
        print ("Images de calibration crées")

    def show_some_images(self):
        print("Affichage des images")

        a = 0
        while a < 150:
            im = cv2.imread(self.folder + "/images/image_" + str(a) + ".jpg")
            cv2.imshow("Original", im)

            if time() - self.t > 1:
                print([a]*10 )
                a += 1
                self.t = time()

            k = cv2.waitKey(10) & 0xFF
            if k == 27:  # ord('q'):
                break
        cv2.destroyAllWindows()
        print ("The End ...")

    def read_chessboards(self):
        """Charuco base pose estimation."""

        print("POSE ESTIMATION starts:")

        images = [self.folder + "/images/" + f for f in os.listdir(self.folder\
                  + "/images") if f.startswith("image_")]

        allCorners = []
        allIds = []
        decimator = 0

        for im in images:
            print("=> Processing image {0}".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            res = cv2.aruco.detectMarkers(gray, self.aruco_dict)

            if len(res[0])>0:
                res2 = cv2.aruco.interpolateCornersCharuco( res[0],
                                                            res[1],
                                                            gray,
                                                            self.board)

                if res2[1] is not None and res2[2] is not None:
                    if len(res2[1]) > 3 and decimator % 1 == 0:
                        allCorners.append(res2[1])
                        allIds.append(res2[2])

            decimator+=1

        imsize = gray.shape
        print("POSE ESTIMATION finished")
        return allCorners, allIds, imsize

    def calibrate_camera(self):
        """
        Calibrates the camera using the dected corners.
        """

        print("CAMERA CALIBRATION start ...")

        allCorners, allIds, imsize = self.read_chessboards()

        print("Calcul start ...")

        cameraMatrixInit = np.array([[ 2000.,    0., imsize[0]/2.],
                                     [    0., 2000., imsize[1]/2.],
                                     [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)

        ret, camera_matrix, distortion_coefficients0, rotation_vectors,\
        translation_vectors, stdDeviationsIntrinsics, stdDeviationsExtrinsics,\
        perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
                                        charucoCorners=allCorners,
                                        charucoIds=allIds,
                                        board=self.board,
                                        imageSize=imsize,
                                        cameraMatrix=cameraMatrixInit,
                                        distCoeffs=distCoeffsInit,
                                        flags=flags,
                                        criteria=(cv2.TERM_CRITERIA_EPS &\
                                                  cv2.TERM_CRITERIA_COUNT,
                                                  10000,
                                                  1e-9)
                                        )

        print("CAMERA CALIBRATION finished")
        print(ret, camera_matrix, distortion_coefficients0, rotation_vectors,\
               translation_vectors)

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors,\
               translation_vectors

    def calibration_test_init(self):
        ret, mtx, dist, rvecs, tvecs = self.calibrate_camera()
        print("mtx =", mtx)
        print("dist =", dist)

        np.savetxt(self.folder + "/calib_mtx_webcam.csv", mtx)
        np.savetxt(self.folder + "/calib_dist_webcam.csv", dist)

    def calibration_test(self):

        mtx = np.loadtxt(self.folder + "/calib_mtx_webcam.csv")
        dist = np.loadtxt(self.folder + "/calib_dist_webcam.csv")

        i = 24 # select image id
        plt.figure()
        frame = cv2.imread(self.folder + "/images/image_100.jpg")
        img_undist = cv2.undistort(frame, mtx, dist, None)
        plt.subplot(211)
        plt.imshow(frame)
        plt.title("Raw image")
        plt.axis("off")
        plt.subplot(212)
        plt.imshow(img_undist)
        plt.title("Corrected image")
        plt.axis("off")
        plt.show()

    def translation_rotation_test(self):
        """
        tvecs = [[[ 0.13069635 -0.14270095  0.95942093]]

         [[ 0.1985628  -0.19638588  0.97662132]]

         [[ 0.08067213 -0.20132235  0.9697723 ]]

         [[ 0.15022369 -0.2583332   1.00096782]]]

        rvecs = [[[ 2.10555692 -1.86512302  0.02889871]]

         [[ 2.10135963 -1.85605388  0.01595754]]

         [[ 2.1019986  -1.85241087  0.10690299]]

         [[ 2.14865675 -1.86506622 -0.01578507]]]
        """

        frame = cv2.imread(self.folder + "/images/image_10.jpg")

        mtx = np.loadtxt(self.folder + "/calib_mtx_webcam.csv")
        dist = np.loadtxt(self.folder + "/calib_dist_webcam.csv")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          self.aruco_dict,
                                                          parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame, corners, ids)

        # # Result
        conn = np.array([0, 1, 2, 3, 0])
        plt.figure()
        plt.imshow(frame_markers)
        plt.legend()
        plt.show()

        # ### Add local axis on each maker
        size_of_marker =  0.046  #0.0145 # side lenght of the marker in meter
        rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners,
                                                              size_of_marker,
                                                              mtx,
                                                              dist)

        print("tvecs =", tvecs)
        print("rvecs =", rvecs)

        length_of_axis = 0.01
        imaxis = aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(tvecs)):
            imaxis = aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i],
                                    length_of_axis)

        cv2.imwrite(self.folder + '/charuco_test.png', imaxis)

    def detect_one_real_marker(self):

        mtx = np.loadtxt(self.folder + "/calib_mtx_webcam.csv")
        dist = np.loadtxt(self.folder + "/calib_dist_webcam.csv")
        self.sender = Multicast("224.0.0.11", 18888)

        while 1:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            parameters =  aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                                                      gray,
                                                      self.aruco_dict,
                                                      parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(frame, corners, ids)

            # Add local axis on each maker
            size_of_marker =  0.046  # side lenght of the marker in meter
            rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners,
                                                                  size_of_marker,
                                                                  mtx,
                                                                  dist)

            cv2.imshow("World co-ordinate axes", frame_markers)
            cv2.imshow("Reglage", self.reglage_img)

            # ## Send to Blender
            if rvecs is not None and tvecs is not None:
                data = json.dumps({"rvec": rvecs.tolist(),
                                   "tvec": tvecs.tolist()}).encode("utf-8")
                print(data)
                self.sender.send_to(data, ("224.0.0.11", 18888))

            if cv2.waitKey(100) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    cam = 0
    cf = MyConfig("./charuco.ini")
    conf = cf.conf
    apply_all_cam_settings(conf["HD5000"], cam)

    ct = CharucoTest(cam, cf)

    # #ct.create_markers_pdf()
    # #ct.create_chessboard_png()
    # #ct.create_images_calibration()
    # #ct.show_some_images()
    # #ct.read_chessboards()
    # #ct.calibrate_camera()
    # #ct.calibration_test_init()
    # #ct.calibration_test()
    # #ct.translation_rotation_test()
    ct.detect_one_real_marker()


"""
data = pd.DataFrame(data=tvecs.reshape(43, 3), columns=["tx","ty","tz"],
                  index=ids.flatten())
data.index.name = "makers"
data.sort_index(inplace=True)
print("data =", data)

p = data.values
((p[1]-p[0])**2.).sum()**.5,((p[2]-p[1])**2.).sum()**.5,((p[3]-p[2])**2.).sum()**.5

((data.loc[11]-data.loc[0]).values**2).sum()

V0_1= p[1]-p[0]
V0_11=p[11]-p[0]

print("V0_1 =", V0_1, "V0_11 =", V0_11)
print("np.dot(V0_1, V0_11) =", np.dot(V0_1, V0_11))

fig=plt.figure()
ax= fig.add_subplot(1, 1, 1)
ax.set_aspect("equal")
plt.plot(data.tx[:10], data.ty[:10], "or-")
plt.grid()
plt.show()
print("data.tx =", data.tx)

corners = np.array(corners)
# pd = panda
data2 = pd.DataFrame({"px":corners[:,0,0,1],
                      "py":corners[:,0,0,0]},
                       index=ids.flatten())
data2.sort_index(inplace=True)
print("data2 =", data2)

n0 = data2.loc[0]
n1 = data2.loc[1]
d01 = ((n0-n1).values**2).sum()**.5
d = 42.5e-3
factor = d/d01
data2["x"] = data2.px * factor
data2["y"] = data2.py * factor
d1_0 = data2.loc[2].y - data2.loc[1].y
d11_0 = data2.loc[11].x - data2.loc[0].x
print("d1_0 =", d1_0)
print("d11_0 =", d11_0)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(3, 3, .025, .0125, dictionary)
img = board.draw((200*3, 200*3))
cv2.imwrite(self.folder + '/charuco_test.png',img)

help(aruco)
charucoCorners = allCorners, charucoIds=allIds, board=board,
                 imageSize=imsize, cameraMatrix=cameraMatrixInit,
                 distCoeffs=distCoeffsInit, flags=flags,
                 criteria=(cv2.TERM_CRITERIA_EPS &\
                 cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
help (aruco.calibrateCameraCharucoExtended)
"""
