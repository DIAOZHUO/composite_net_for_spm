import os
import SPMUtil as spmu
import numpy as np
import torch
import matplotlib.pyplot as plt
from enum import Enum, IntEnum
from nets.topography_composite_net.predict import *




if __name__ == '__main__':
    path = "./878_si_20230606.pkl"


    net1_output, net2_output, net3_output = predict_by_path(path)
    data = spmu.DataSerializer(path)
    data.load()
    map = spmu.flatten_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name])
    # print(net1_output)
    # print(net2_output)
    # print(net3_output)

    """
    net1 result
    """
    print("index", int(net1_output[0]))
    print("net1 raw output", net1_output[4])
    tip_quality = TipQualityType(int(net1_output[0]))
    text = "tip type - " + tip_quality.name + " in " + str(net1_output[1]*100) + "%, "
    if net1_output[2]:
        print(text+"bad good judge in " + str(net1_output[3]*100) + "%")
    else:
        print("not clear tip", text)

    """
    net2 plot
    """
    for it in net2_output:
        if it[5] == 0:
            spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "cyan", linewidth=3, linestyle="dashed")
        elif it[5] == 1:
            spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "r", linewidth=3, linestyle="dashed")

    plt.imshow(map, cmap="afmhot")
    plt.axis("off")
    plt.show()
    plt.clf()
    """
    net3 plot
    """

    plt.imshow(map, cmap="afmhot")
    plt.axis("off")

    boxes, kps = net3_output
    kp_threshold = 0.3
    for i, box in enumerate(boxes):
        # if box[5] == 1:
        #     continue
        if box[5] == 0:
            spmu.Rect2D((box[0], box[1]), box[2] - box[0], box[3] - box[1]).draw_rect_patch_on_matplot(plt.gca(), c="yellow", linestyle="dashed")
        elif box[5] == 1:
            spmu.Rect2D((box[0], box[1]), box[2] - box[0], box[3] - box[1]).draw_rect_patch_on_matplot(plt.gca(), c="magenta", linestyle="dashed")
        pts = kps[i]
        for j, pt in enumerate(pts[:3]):
            if pt[2] > kp_threshold and 10 < pt[0] < map.shape[1] - 10 and 10 < pt[1] < map.shape[0] - 10:
                plt.gca().scatter(pt[0], pt[1], c="r")

        atom_color = "yellow" if box[5] == 0 else "magenta"
        for j, pt in enumerate(pts[3:]):
            if pt[2] > kp_threshold and 10 < pt[0] < map.shape[1] - 10 and 10 < pt[1] < map.shape[0] - 10:
                plt.gca().scatter(pt[0], pt[1], c=atom_color)


    plt.show()

