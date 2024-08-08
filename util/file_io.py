import os
import json
import cv2
import matplotlib.pyplot as plt
import SPMUtil as spmu
import numpy as np
from typing import List
from pathlib import Path
import re


def get_datasets_filename(folder_path, ext="png"):
    file_list = []
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    for fileName in files:
        fileName, extension = os.path.splitext(fileName)
        if extension == ".png":
            file_list.append(fileName)
    return file_list




def find_file_from_walk(walk_path, condition=lambda c: True) -> List[spmu.DataSerializer]:
    file_list = []
    for curdir, dirs, files in os.walk(walk_path):
        for fileName in files:
            fileName, extension = os.path.splitext(fileName)
            if extension == ".pkl":
                try:
                    dataSerializer = spmu.DataSerializer(os.path.join(curdir, fileName))
                    dataSerializer.load()
                    if condition(dataSerializer):
                        file_list.append(dataSerializer)
                except:
                    print("error occur while loading", os.path.join(curdir, fileName))

    return file_list

def find_file_from_folder(folder_path, condition=lambda c: True) -> List[spmu.DataSerializer]:
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s)]


    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    files.sort(key=alphanum_key)
    file_list = []
    for fileName in files:
        fileName, extension = os.path.splitext(fileName)
        if extension == ".pkl":
            try:
                dataSerializer = spmu.DataSerializer(os.path.join(folder_path, fileName))
                dataSerializer.load()
                if condition(dataSerializer):
                    file_list.append(dataSerializer)
            except Exception as e:
                print("error occur while loading", os.path.join(folder_path, fileName), e)

    return file_list


def convert_wsxm(data: spmu.DataSerializer):
    if spmu.cache_2d_scope.FWFW_ZMap.name in data.data_dict.keys():

        scan_param=spmu.PythonScanParam.from_dataSerilizer(data)
        map=data.data_dict["FWFW_ZMap"]

        x_piezo, y_piezo, z_piezo = 1, 1, 1
        if 'HardwareConfigure' in data.data_dict:
            hardware = json.loads((data.data_dict['HardwareConfigure']))
            if 'Tube_Scanner_X_Piezo_Calibration' in hardware:
                x_piezo = hardware['Tube_Scanner_X_Piezo_Calibration']
            if 'Tube_Scanner_Y_Piezo_Calibration' in hardware:
                y_piezo = hardware['Tube_Scanner_Y_Piezo_Calibration']
            if 'Tube_Scanner_Z_Piezo_Calibration' in hardware:
                z_piezo = hardware['Tube_Scanner_Z_Piezo_Calibration']

        file_name = os.path.basename(data.path).split(".")
        file_name_wsxm = file_name[0]+".txt"
        with open(file_name_wsxm, "w", encoding='UTF-8',newline="\r\n") as f:
            f.write("WSxM file copyright UAM\n")
        with open(file_name_wsxm, "a", encoding='UTF-8',newline="\r\n") as f:
            f.write("WSxM ASCII Matrix file\n")
            f.write("X Amplitude: %snm \n"%str((scan_param.Aux1MaxVoltage-scan_param.Aux1MinVoltage) * x_piezo))
            f.write("Y Amplitude: %snm \n"%str((scan_param.Aux1MaxVoltage-scan_param.Aux1MinVoltage) * y_piezo))
            f.write("Z Amplitude: %s nm\n" % str(z_piezo))
            np.savetxt(f, map, delimiter="\t")



def get_scan_file_by_order(path, order=1):
    index = int(os.path.basename(path).split("_")[0]) + order
    index_str = "{:0>3}".format(str(index))
    base = index_str + "_" + "_".join((os.path.basename(path).split("_")[1:]))
    f_path = os.path.join(Path(path).parent, base) + ".pkl"
    if os.path.exists(f_path):
        return f_path
    else:
        return None







if __name__ == '__main__':
    datasets = get_datasets_filename("E:\PythonProjects\yolov8\datasets\kp_detection/dataset1") + get_datasets_filename("E:\PythonProjects\yolov8\datasets\kp_detection/dataset2")
    for it in get_datasets_filename("E:\PythonProjects\yolov8\datasets\kp_detection/add2"):
        print(it)

