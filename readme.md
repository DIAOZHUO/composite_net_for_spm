## AI-equipped scanning probe microscopy for autonomous site-specific atomic-level characterization at room temperature


The upload contains code of our AI models, and produce the figure in our paper.
Here is the directory instruction.

- datas: the datas for sts measurement (figure 4, 5)
- figs: the code to produce the figure
- nets: the implement of our AI model. 
- util: some utility functions used for data IO and processing.



### Installation

The code is tested on Windows 10/11, GeForce RTX 4090.
We runs the code in python 3.11. 
It is recommended to use conda to setup python environment. 

```bash
conda create -n your_project_name python=3.11
```


Install cuda and pytorch. The pytorch version should match your cuda version.
https://pytorch.org/get-started/locally/.
In our case, 

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the other requirement package.

```bash
pip install -r requirements.txt
```


