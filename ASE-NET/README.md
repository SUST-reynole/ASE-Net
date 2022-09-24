# Semi-supervised medical image segmentation using adversarial consistency learning and dynamic convolution network



### Usage
1. Clone the repo.;
```
git clone https://github.com/SUST-reynole/ASE-Net.git
```
2. Put the data in './data';

3. Train the model;
```
cd ASE-Net
# e.g., for 20% labels on LA
python3 ./code/train_ASE-Net_3D.py --dataset_name LA  --labelnum 16 --gpu 0 --temperature 0.1
```
4. Test the model;
```
cd ASE-Net
# e.g., for 20% labels on LA
python ./code/test_LA_our.py --dataset_name LA  --labelnum 16 --gpu 0
```

### Acknowledgements:
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MC-Net](https://github.com/ycwu1997/MC-Net). Thanks for these authors for their valuable works and hope our model ASE-Net can promote the relevant research as well.

### Questions

