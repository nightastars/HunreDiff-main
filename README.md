# HunreDiff-main
Hybrid-domain Collaborative Universal Recursive Diffusion Model for Non-ideal Measurement CT Reconstruction under Extreme Degradation Conditions

If you want to train the network, run train_diffusion.py and modify the parameter configurations (configs/CT.yml); if you want to perform inference, run eval_diffusion.py and modify the parameter configurations (configs/CT.yml). In the file models/restorantion.py, parameters for CT reconstruction can be modified, including forward projection, FBP, and SIRT.

Please pay attention to the dataset format. For specific format requirements, refer to dataset/CTdata.py.
