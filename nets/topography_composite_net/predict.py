import os
import SPMUtil as spmu
import numpy as np
import torch
import matplotlib.pyplot as plt
from enum import Enum, IntEnum
import nets.topography_composite_net.check_model as check_model


class TipQualityType(IntEnum):
    Good_Best = 0
    Good = 1
    Good_Fat = 2
    Good_Multi = 3
    Good_bad_area = 4
    Good_TipChange = 5
    Step = 6
    Bad_Noisy = 7
    Bad_TipChange = 8
    Bad = 9
    Bad_Area = 10





model = check_model.get_model()
# model = CompositeNet()
# model.load(str(Path(__file__).parent.absolute()) + "./composite_net.pt")
model.eval()
# summary(model.Net2.model, [(3, 256, 256)], device="cpu")



def predict_by_data(data, net1_threshold=3.2, net2_box_threshold=0.15, net3_box_threshold=0.5):
    with torch.no_grad():
        if isinstance(data, spmu.DataSerializer):
            data.load()
            results = model(process_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name]))
        elif isinstance(data, np.ndarray):
            results = model(data)
        else:
            raise ValueError("unsupported input data type", type(data))

        # Postprocess
        net1_output = _process_net1_predict_result(results[0], threshold=net1_threshold)
        net2_output = _process_net2_predict_result(results[1], threshold=net2_box_threshold)
        net3_output = _process_net3_predict_result(results[2], box_threshold=net3_box_threshold)

    return net1_output, net2_output, net3_output


def predict_by_path(path, net1_threshold=3.2, net2_box_threshold=0.15, net3_box_threshold=0.5):
    filename, ext = os.path.splitext(path)
    if ext == ".pkl":
        data = spmu.DataSerializer(path)
        return predict_by_data(data, net1_threshold, net2_box_threshold, net3_box_threshold)
    else:
        with torch.no_grad():
            results = model(path)

        # Postprocess
        net1_output = _process_net1_predict_result(results[0], threshold=net1_threshold)
        net2_output = _process_net2_predict_result(results[1], threshold=net2_box_threshold)
        net3_output = _process_net3_predict_result(results[2], box_threshold=net3_box_threshold)

        return net1_output, net2_output, net3_output





def process_map(map):
    map = spmu.flatten_map(map, spmu.FlattenMode.Average)
    map = spmu.filter_2d.gaussian_filter(map, 1)
    map = spmu.formula.normalize01_2dmap(map) * 255
    map = np.stack((map,) * 3, axis=-1)
    return map


def _process_net1_predict_result(results, threshold=3.2):
    """
    :param results:
    :param threshold:
    :return: index result, index possibilty weight, is good tip result, is good tip? possibilty weight, raw output value
    """
    index = torch.max(results.data, 1)[1].item()
    result = results.to('cpu').detach().numpy()[0].copy()
    # print(index, result)
    if index == 6:
        return index, result[6] / np.sum(result), False, 0, result
    # remove step tip: 15 elements with step 0
    result_list = []
    for i in range(result.shape[0]):
        if result[i] < threshold or i == 6:
            result_list.append(0)
        else:
            result_list.append(result[i])
    result_list = np.array(result_list)

    if sum(result_list) == 0:
        return index, 0, False, 0, result
    good_rate = np.sum(result_list[:6]) / np.sum(result_list)
    if (index < 6) != (good_rate > 0.5):
        return index, result[index] / np.sum(result), False, 0, result
    if index < 6:
        return index, result_list[index] / np.sum(result_list), True, np.sum(result_list[:6]) / np.sum(result_list), result
    else:
        return index, result_list[index] / np.sum(result_list), True, np.sum(result_list[6:]) / np.sum(result_list), result



def _process_net2_predict_result(results, threshold=0.3):
    result = results[0].boxes.data.to('cpu').detach().numpy().copy()
    output = []
    for it in result:
        if it[4] > threshold:
            output.append(it)
    return output


def _process_net3_predict_result(results, box_threshold=0.5):
    boxes = results[0].boxes.data.to('cpu').detach().numpy().copy()
    kps = results[0].keypoints.data.to('cpu').detach().numpy().copy()

    box_output = []
    kp_output = []
    shape = results[0].orig_shape
    for i, box in enumerate(boxes):
        center_pt = (box[0]+box[2])/2, (box[1]+box[3])/2
        if box[4] > box_threshold and 10 < center_pt[0] < shape[1] - 10 and 10 < center_pt[1] < shape[0] - 10:
            box_output.append(box)
            kp_output.append(kps[i])

    return box_output, kp_output






