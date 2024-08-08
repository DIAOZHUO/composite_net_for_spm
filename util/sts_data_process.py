import SPMUtil as spmu
import numpy as np
from numpy import dot
from numpy.linalg import norm

from scipy import interpolate





def filter_1d(ar):
    k = np.ones(31)/31
    ar = spmu.filter_1d.savgol_filter(ar, 15, 2)
    ar = np.convolve(ar, k, mode="valid")
    return ar

def get_iv_data(data: spmu.DataSerializer, y_value_limiter=(-10, 10)):
    data.load()
    stage = spmu.StageConfigure.from_dataSerilizer(data)
    # print(data.data_dict.keys())
    print(data.data_dict['addition'])


    i = data.data_dict['ArrayBuilderParam']['scan_result_i']
    i_out = i[1]
    left_margin = len(i[0])
    right_margin = len(i[2])

    bias_offset = 0.002
    v = data.data_dict['ArrayBuilderParam']["scan_info"]["scan_array_v"][left_margin:-right_margin] + stage.Sample_Bias + bias_offset

    data_length = int(len(i_out) / 2)
    i_out_1 = (filter_1d(i_out[:data_length]) + bias_offset) * 10
    i_out_2 = (filter_1d(i_out[data_length:][::-1]) + bias_offset) * 10
    if i_out_1[0] < y_value_limiter[0] or i_out_2[0] < y_value_limiter[0]:
        return None
    if i_out_1[-1] > y_value_limiter[1] or i_out_2[-1] > y_value_limiter[1]:
        return None

    v_1 = v[:data_length]
    v_2 = v[data_length:][::-1]

    left_slice = 1
    right_slice = 1
    return v_1[15:-15][::-1][left_slice:-right_slice],\
        (i_out_1[::-1] + bias_offset)[left_slice:-right_slice], \
        v_2[15:-15][::-1][left_slice:-right_slice], \
        (i_out_2[::-1] + bias_offset)[left_slice:-right_slice]


def calc_ldos(v, i):
    mean_size = 11
    v = v[int((mean_size-1)/2):-int((mean_size-1)/2)]
    k = np.ones(mean_size)/mean_size
    i = np.convolve(i, k, mode="valid")
    didv = np.diff(i) / (v[1]-v[0])

    mean_size = 31
    k = np.ones(mean_size)/mean_size
    didv = spmu.filter_1d.savgol_filter(didv, 15, 2)
    didv = np.convolve(didv, k, mode="valid")

    for k in range(len(didv)):
        if abs(v[k]) > 0.05:
            didv[k] = didv[k] / (i[k]/v[k])
        else:
            didv[k] = np.nan

    valid = np.nonzero(~np.isnan(didv))
    didv = interpolate.interp1d(v[int((mean_size-1)/2):-int((mean_size-1)/2)][valid].tolist(), didv[valid].tolist(), kind='cubic', bounds_error=False)(v[int((mean_size-1)/2):-int((mean_size-1)/2)])
    v = v[int((mean_size-1)/2):-int((mean_size-1)/2)]
    return v, didv


def calc_conductance(v, i):
    v_list = []
    g = []
    for k, it in enumerate(v):
        if it != 0:
            v_list.append(it)
            g.append(i[k]/v[k])
    return np.array(v_list), np.array(g)


def calc_cos_sim(i1, i2):
    cos_sim = dot(i1, i2) / (norm(i1) * norm(i2))
    return cos_sim


def calc_cos_sim_list(i_list, data_slice=0):
    v = []
    for i in range(len(i_list)):
        value = 0
        for j in range(len(i_list)):
            # if i==j:
            #     continue
            value += calc_cos_sim(i_list[i][data_slice:], i_list[j][data_slice:])
        value /= len(i_list)
        v.append(value)
    return v


def find_sim_file_from_index(data_list, index, th_sim=0.9):
    l = []
    for i, it in enumerate(data_list):
        if i == index:
            continue
        sim = calc_cos_sim(data_list[index], it)
        if sim > th_sim:
            l.append(it)
    return l



