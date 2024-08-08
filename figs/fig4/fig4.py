from __future__ import annotations
import SPMUtil as spmu
import matplotlib.pyplot as plt
from matplotlib import cm
from util.draw_scan_area import ScanAreaManager
from util.file_io import find_file_from_folder



def topo_condition(data: spmu.DataSerializer):
    header = spmu.ScanDataHeader.from_dataSerilizer(data)
    param = spmu.PythonScanParam.from_dataSerilizer(data)
    if header.Array_Builder == "Common2DScanArrayBuilder" and param.Aux1MaxVoltage - param.Aux1MinVoltage < 0.4:
        return True
    return False



if __name__ == '__main__':
    datas = find_file_from_folder("../../datas/si_20230506", condition=topo_condition)[25:]
    s = ScanAreaManager()
    img, x, y = s.UpdateBackgroundTexture(datas)

    img = img[3720:4455, 10172:12800]

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(img, cmap="afmhot")
    axes[0].plot(x - 10172, y - 3720, c="red", marker="v")

    axes[0].set_xticks([])
    axes[0].set_yticks([])

    state, x, y = s.UpdateTipFlagTexture(datas)
    state = state[3720:4455, 10172:12800]

    cmap = cm.get_cmap('tab10')
    cmap.set_under('black')
    axes[1].imshow(state, cmap=cmap, vmin=.001)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.show()


