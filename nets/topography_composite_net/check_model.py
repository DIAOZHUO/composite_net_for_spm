import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from .model import CompositeNet
import gdown


def get_model():
    model = CompositeNet()
    if os.path.isfile(str(Path(__file__).parent.absolute()) + "./composite_net.pt"):
        model.load(str(Path(__file__).parent.absolute()) + "./composite_net.pt")
        print("     Topography Composite Network Model Load Finished!")
    else:
        print("     downloading Topography Composite Network Model...")
        gdown.download(id="1nJH8Ut1k7lT-qG7OYRTCixxmr54-ZDEw", output=str(Path(__file__).parent.absolute()) + "./composite_net.pt")
        # download_file_from_google_drive(id='1nJH8Ut1k7lT-qG7OYRTCixxmr54-ZDEw',
        #                                 destination=str(Path(__file__).parent.absolute()) + "./composite_net.pt")
        model.load(str(Path(__file__).parent.absolute()) + "./composite_net.pt")
        print("     Topography Composite Network Model Load Finished!")
    return model


if __name__ == '__main__':
    get_model()
