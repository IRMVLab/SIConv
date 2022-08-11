## ScanNet v2 Data

Please download original dataset from weibsite: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

To prepare the Scannet dataset for training and evaluation, modity [line 83](https://github.com/IRMVLab/SIConv/blob/e2609e4cb97ab14a4a8758edf04712a410df1dfd/scannet/scannetv2_seg_dataset_rgb21c_pointid.py#L83) in `scannetv2_seg_dataset_rgb21c_pointid.py` to your ScanNet v2 dataset path.

Then,

```
python scannetv2_seg_dataset_rgb21c_pointid.py
```

This will generate three pickle files: `scannet_train_rgb21c_pointid.pickle`, `scannet_val_rgb21c_pointid.pickle`, and `scannet_test_rgb21c_pointid.pickle`. The first two are used in training and validation.
