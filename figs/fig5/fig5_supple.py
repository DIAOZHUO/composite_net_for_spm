import SPMUtil as spmu
import numpy as np
import matplotlib.pyplot as plt
from util.file_io import find_file_from_folder
from util.sts_data_process import *
import json
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import scipy
import os
from scipy import interpolate
import pandas as pd
from scipy.interpolate import UnivariateSpline



def condition(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            return True
    return False


def condition_adsorp(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            sts_param = json.loads(data.data_dict['addition'])["STSParam"]
            if sts_param['obj_type'] == 2:
                return True
    return False


def condition_ch(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            sts_param = json.loads(data.data_dict['addition'])["STSParam"]
            if sts_param['obj_type'] == 0 or sts_param['obj_type'] == 1:
                if sts_param['kp_type'] == 0 or sts_param['kp_type'] == 1 or sts_param['kp_type'] == 2:
                    return True
    return False


def condition_uc1_ce(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            sts_param = json.loads(data.data_dict['addition'])["STSParam"]
            if sts_param['obj_type'] == 0:
                if sts_param['kp_type'] == 4 or sts_param['kp_type'] == 5 or sts_param['kp_type'] == 6:
                    return True
    return False


def condition_uc1_co(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            sts_param = json.loads(data.data_dict['addition'])["STSParam"]
            if sts_param['obj_type'] == 0:
                if sts_param['kp_type'] == 3 or sts_param['kp_type'] == 7 or sts_param['kp_type'] == 8:
                    return True
    return False

def condition_uc2_ce(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            sts_param = json.loads(data.data_dict['addition'])["STSParam"]
            if sts_param['obj_type'] == 1:
                if sts_param['kp_type'] == 4 or sts_param['kp_type'] == 5 or sts_param['kp_type'] == 6:
                    return True
    return False


def condition_uc2_co(data: spmu.DataSerializer):
    if 'addition' in data.data_dict.keys():
        if "STSParam" in data.data_dict['addition']:
            sts_param = json.loads(data.data_dict['addition'])["STSParam"]
            if sts_param['obj_type'] == 1:
                if sts_param['kp_type'] == 3 or sts_param['kp_type'] == 7 or sts_param['kp_type'] == 8:
                    return True
    return False


"""
"""



def plot_weight(condition, th_sim, ax):
    data_list = find_file_from_folder("../../datas/si_20230719_roists", condition)
    data_list += find_file_from_folder("../../datas/si_20230720_roists", condition)
    data_list += find_file_from_folder("../../datas/si_20230721_roists", condition)

    i_list = []
    v_list = []
    name_list = []

    for it in data_list:
        data = get_iv_data(it)
        if data is None:
            continue
        v_1, i_out_1, v_2, i_out_2 = data
        sim = calc_cos_sim(i_out_1, i_out_2)
        if sim > 0.86:
            i_list.append(i_out_2)
            v_list.append(v_2)
            name_list.append(it.path + "-2")






    sim_list = calc_cos_sim_list(i_list, data_slice=0)
    print("best idx", np.argmax(sim_list), np.max(sim_list))
    print("best data", name_list[np.argmax(sim_list)])

    # sns.histplot(sim_list, bins=30)
    # plt.axvline(x=th_sim, c="r")
    # plt.show()

    good_count = 0
    bad_count = 0
    i_mean = np.zeros(i_list[0].shape)





    for i, it in enumerate(v_list):
        v, didv = calc_ldos(v_list[0], i_list[i])
        spl_didv = UnivariateSpline(v[:-1], didv[:-1], k=3, s=5)
        didv = spl_didv(v)

        if sim_list[i] > th_sim:
            i_mean += i_list[i]
            good_count += 1
            ax.plot(v, didv, c="red", alpha=0.02)

        else:
            bad_count += 1
            ax.plot(v, didv, c="black", alpha=0.01)


    i_mean /= good_count

    print("good count", good_count)
    print("bad count", bad_count)
    v, didv = calc_ldos(v_list[0], i_mean)


    spl_sts = UnivariateSpline(v[:-1], didv[:-1], k=3, s=5)
    ax.plot(v, spl_sts(v), c="blue", ls="--", lw=1.5)
    ax.set_ylim(0, 4)




fig, axes = plt.subplots(2,2)
plot_weight(condition_uc1_co, 0.9685, axes[0,0])
plot_weight(condition_uc1_ce, 0.9685, axes[0,1])
plot_weight(condition_uc2_co, 0.9718, axes[1,0])
plot_weight(condition_uc2_ce, 0.9718, axes[1,1])
plt.show()

