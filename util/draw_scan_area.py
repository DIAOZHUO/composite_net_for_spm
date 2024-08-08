from __future__ import annotations
import os.path

from typing import List
import numpy as np
import SPMUtil as spmu
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import nets.topography_composite_net.predict as predict
from util.file_io import find_file_from_folder



class ScanAreaManager:
    BGTEXMAXSIZE = (12800, 12800)

    def __init__(self):
        super().__init__()

        self.area_background_textrue = np.zeros(shape=self.BGTEXMAXSIZE, dtype=np.uint8)
        self.area_background_max_texture_count = 10000
        self.visible_scan_area_zoom_rate = 1.75

        self.visible_scan_area = self.ScanAreaMax

    @staticmethod
    def normalize01_2dmap(array):
        min, max = np.min(array), np.max(array)
        return (array - min) / (max - min)



    def UpdateBackgroundTexture(self, files: List[spmu.DataSerializer]):
        # print("UpdateBackgroundTexture")

        length = len(files)
        tmp = files
        start_index = max(length - self.area_background_max_texture_count, 0)
        bg = np.zeros(self.BGTEXMAXSIZE, dtype=np.uint8)
        x_coord = []
        y_coord = []

        try:
            for i in range(start_index, length):
                param = spmu.PythonScanParam.from_dataSerilizer(tmp[i])
                stage = spmu.StageConfigure.from_dataSerilizer(tmp[i])

                if spmu.cache_2d_scope.FWFW_ZMap.name in tmp[i].data_dict and param.Aux1Type == "X" and param.Aux2Type == "Y":
                    map = spmu.flatten_map(tmp[i].data_dict[spmu.cache_2d_scope.FWFW_ZMap.name].copy())
                    if len(map.shape) == 2:
                        # print("load texture", tmp[i].path)
                        resize_ratio = np.abs(param.Aux2MaxVoltage - param.Aux2MinVoltage) / self.visible_scan_area.Height, \
                                       np.abs(param.Aux1MaxVoltage - param.Aux1MinVoltage) / self.visible_scan_area.Width
                        map = (self.normalize01_2dmap(map) * 254).astype(np.uint8) + 1
                        # map = cv2.imread('./lena.jpg')[:,:,0]
                        # map = cv2.resize(map, dsize=(param.Aux2ScanSize, param.Aux1ScanSize), interpolation=cv2.INTER_CUBIC)
                        # map = (self.normalize01_2dmap(map) * 254).astype(np.uint8) + 1

                        # aux1:x, aux2:y only
                        size = (int(self.BGTEXMAXSIZE[0] *resize_ratio[0]), int(self.BGTEXMAXSIZE[1] *resize_ratio[1]))
                        resize_map = cv2.resize(map, dsize=size, interpolation=cv2.INTER_CUBIC)
                        x, y = self.get_xy_signal_offset(param, stage)


                        pts = int((y - self.visible_scan_area.y1) / self.visible_scan_area.Height * self.BGTEXMAXSIZE[0]), \
                            int((x - self.visible_scan_area.x1) / self.visible_scan_area.Width * self.BGTEXMAXSIZE[1])

                        # print(pts)
                        if pts[0] + size[0] > 0 and pts[1] + size[1] > 0 \
                                and pts[0] < self.BGTEXMAXSIZE[0] and pts[1] < self.BGTEXMAXSIZE[1]:
                            x_coord.append(pts[1])
                            y_coord.append(pts[0])

                            x1 = max(-pts[1], 0)
                            if pts[1] + size[1] > self.BGTEXMAXSIZE[1]:
                                x2 = self.BGTEXMAXSIZE[1 ] -pts[1 ] -size[1]
                            else:
                                x2 = size[1]
                            y1 = max(-pts[0], 0)
                            if pts[0] + size[0] > self.BGTEXMAXSIZE[0]:
                                y2 = self.BGTEXMAXSIZE[0 ] -pts[0 ] -size[0]
                            else:
                                y2 = size[0]
                            # print(max(pts[0], 0), pts[0]+size[0], max(pts[1], 0), pts[1]+size[1])
                            bg[max(pts[0], 0):pts[0 ] +size[0], max(pts[1], 0):pts[1 ] +size[1]] = resize_map[y1:y2, x1:x2]

                            # text_position = (np.clip(pts, 10, self.BGTEXMAXSIZE[0 ] -10 ) -5)
                            # bg_text = np.zeros(self.BGTEXMAXSIZE, dtype=np.uint8)
                            # cv2.putText(bg_text, os.path.basename(tmp[i].path), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150), 1, cv2.LINE_AA)
                            # bg_text = np.flip(bg_text, axis=0)
                            # bg = cv2.bitwise_or(bg, bg_text)
        except:
            pass
        self.area_background_textrue = bg
        return bg, np.array(x_coord), np.array(y_coord)


    def UpdateTipFlagTexture(self, files: List[spmu.DataSerializer]):
        # print("UpdateBackgroundTexture")

        length = len(files)
        tmp = files
        start_index = max(length - self.area_background_max_texture_count, 0)
        # bg = np.empty(self.BGTEXMAXSIZE, dtype=np.uint8)
        bg = np.full(self.BGTEXMAXSIZE, np.nan)
        x_coord = []
        y_coord = []

        try:
            for i in range(start_index, length):
                param = spmu.PythonScanParam.from_dataSerilizer(tmp[i])
                stage = spmu.StageConfigure.from_dataSerilizer(tmp[i])

                if spmu.cache_2d_scope.FWFW_ZMap.name in tmp[i].data_dict and param.Aux1Type == "X" and param.Aux2Type == "Y":
                    tip_state = predict.predict_by_data(tmp[i], net1_threshold=2.4)[0]
                    print(tip_state)
                    tip_state = tip_state[0]
                    # if tip_state == 0:
                    #     tip_state = None
                    map = np.ones(tmp[i].data_dict[spmu.cache_2d_scope.FWFW_ZMap.name].shape) * tip_state
                    if len(map.shape) == 2:
                        # print("load texture", tmp[i].path)
                        resize_ratio = np.abs(param.Aux2MaxVoltage - param.Aux2MinVoltage) / self.visible_scan_area.Height, \
                                       np.abs(param.Aux1MaxVoltage - param.Aux1MinVoltage) / self.visible_scan_area.Width
                        # map = cv2.imread('./lena.jpg')[:,:,0]
                        # map = cv2.resize(map, dsize=(param.Aux2ScanSize, param.Aux1ScanSize), interpolation=cv2.INTER_CUBIC)
                        # map = (self.normalize01_2dmap(map) * 254).astype(np.uint8) + 1

                        # aux1:x, aux2:y only
                        size = (int(self.BGTEXMAXSIZE[0] *resize_ratio[0]), int(self.BGTEXMAXSIZE[1] *resize_ratio[1]))
                        resize_map = cv2.resize(map, dsize=size, interpolation=cv2.INTER_CUBIC)
                        x, y = self.get_xy_signal_offset(param, stage)


                        pts = int((y - self.visible_scan_area.y1) / self.visible_scan_area.Height * self.BGTEXMAXSIZE[0]), \
                            int((x - self.visible_scan_area.x1) / self.visible_scan_area.Width * self.BGTEXMAXSIZE[1])

                        # print(pts)
                        if pts[0] + size[0] > 0 and pts[1] + size[1] > 0 \
                                and pts[0] < self.BGTEXMAXSIZE[0] and pts[1] < self.BGTEXMAXSIZE[1]:
                            x_coord.append(pts[1])
                            y_coord.append(pts[0])

                            x1 = max(-pts[1], 0)
                            if pts[1] + size[1] > self.BGTEXMAXSIZE[1]:
                                x2 = self.BGTEXMAXSIZE[1 ] -pts[1 ] -size[1]
                            else:
                                x2 = size[1]
                            y1 = max(-pts[0], 0)
                            if pts[0] + size[0] > self.BGTEXMAXSIZE[0]:
                                y2 = self.BGTEXMAXSIZE[0 ] -pts[0 ] -size[0]
                            else:
                                y2 = size[0]
                            # print(max(pts[0], 0), pts[0]+size[0], max(pts[1], 0), pts[1]+size[1])
                            bg[max(pts[0], 0):pts[0 ] +size[0], max(pts[1], 0):pts[1 ] +size[1]] = resize_map[y1:y2, x1:x2]

                            # text_position = (np.clip(pts, 10, self.BGTEXMAXSIZE[0 ] -10 ) -5)
                            # bg_text = np.zeros(self.BGTEXMAXSIZE, dtype=np.uint8)
                            # cv2.putText(bg_text, os.path.basename(tmp[i].path), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150), 1, cv2.LINE_AA)
                            # bg_text = np.flip(bg_text, axis=0)
                            # bg = cv2.bitwise_or(bg, bg_text)
        except:
            pass

        return bg, np.array(x_coord), np.array(y_coord)


    @staticmethod
    def get_xy_signal_offset(param: spmu.PythonScanParam, stage: spmu.StageConfigure):
        x_offset, y_offset = param.XOffset, param.YOffset
        if stage.XY_Scan_Option == 0 or stage.XY_Scan_Option == "Tube Scanner":
            x_offset += stage.Tube_Scanner_Offset_X
            y_offset += stage.Tube_Scanner_Offset_Y
        elif stage.XY_Scan_Option == 1 or stage.XY_Scan_Option == "HS Scanner":
            x_offset += stage.High_Speed_Scanner_Offset_X
            y_offset += stage.High_Speed_Scanner_Offset_Y
        if param.Aux1Type == "X":
            x_offset += param.Aux1MinVoltage
        elif param.Aux1Type == "Y":
            y_offset += param.Aux1MinVoltage
        if param.Aux2Type == "X":
            x_offset += param.Aux2MinVoltage
        elif param.Aux2Type == "Y":
            y_offset += param.Aux2MinVoltage
        return x_offset, y_offset

    @property
    def ScanAreaMax(self) -> ScanVoltageRange:
        max_scan_area = ScanVoltageRange()
        max_scan_area.x1 = -10
        max_scan_area.x2 = 10
        max_scan_area.y1 = -10
        max_scan_area.y2 = 10
        return max_scan_area



class ScanVoltageRange:
    def __init__(self, x1=0.0, y1=1.0, x2=0.0, y2=1.0):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    @property
    def Width(self):
        return self.x2 - self.x1

    @property
    def Height(self):
        return self.y2 - self.y1

    def __str__(self):
        return str(self.x1) + "," + str(self.y1) + "," + str(self.x2) + "," + str(self.y2)

    def Rotate(self, rot):
        rot = rot / 180 * np.pi
        cosa = np.cos(rot)
        sina = np.sin(rot)
        cx, cy = (self.x1+self.x2)/2, (self.y1+self.y2)/2
        # x*cosa-y*sina+cx-cx*cosa+cy*sina
        # x*sina+y*cosa+cy-cx*sina-cy*cosa
        x1 = self.x1*cosa-self.y1*sina+cx-cx*cosa+cy*sina
        y1 = self.x1*sina+self.y1*cosa+cy-cx*sina-cy*cosa
        x2 = self.x2*cosa-self.y2*sina+cx-cx*cosa+cy*sina
        y2 = self.x2*sina+self.y2*cosa+cy-cx*sina-cy*cosa
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __add__(self, other):
        if len(other) == 2 and type(other[0]) is float and type(other[1]) is float:
            return ScanVoltageRange(x1=self.x1+other[0], y1=self.y1+other[1], x2=self.x2+other[0], y2=self.y2+other[1])
        else:
            raise ValueError()


    def __sub__(self, other):
        if len(other) == 2 and type(other[0]) is float and type(other[1]) is float:
            return ScanVoltageRange(x1=self.x1-other[0], y1=self.y1-other[1], x2=self.x2-other[0], y2=self.y2-other[1])
        else:
            raise ValueError()


    def __mul__(self, other):
        if type(other) is float or type(other) is int:
            x_box, ybox = self.x2 - self.x1, self.y2 - self.y1
            _x2, _y2 = self.x1 + x_box * other, self.y1 + ybox * other
            return ScanVoltageRange(x1=self.x1, x2=_x2, y1=self.y1, y2=_y2)
        else:
            raise ValueError()

    def __truediv__(self, other):
        if type(other) is float or type(other) is int:
            x_box, ybox = self.x2 - self.x1, self.y2 - self.y1
            _x2, _y2 = self.x1 + x_box / other, self.y1 + ybox / other
            return ScanVoltageRange(x1=self.x1, x2=_x2, y1=self.y1, y2=_y2)
        else:
            raise ValueError()


    def ToVisibleMarker(self, visible_scan_area):
        x1 = (self.x1 - visible_scan_area.x1) / (visible_scan_area.x2 - visible_scan_area.x1)
        x2 = (self.x2 - visible_scan_area.x1) / (visible_scan_area.x2 - visible_scan_area.x1)
        y1 = (self.y1 - visible_scan_area.y1) / (visible_scan_area.y2 - visible_scan_area.y1)
        y2 = (self.y2 - visible_scan_area.y1) / (visible_scan_area.y2 - visible_scan_area.y1)
        return x1, y1, x2, y2







def topo_condition(data: spmu.DataSerializer):
    header = spmu.ScanDataHeader.from_dataSerilizer(data)
    param = spmu.PythonScanParam.from_dataSerilizer(data)
    if header.Array_Builder == "Common2DScanArrayBuilder" and param.Aux1MaxVoltage - param.Aux1MinVoltage < 0.61:
        return True
    return False

