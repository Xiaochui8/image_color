# image_color



* dataset

需要先有image数据，数据地址在[CelebAMask-HQ-512](https://www.kaggle.com/datasets/vincenttamml/celebamaskhq512)

```cmd
cd data
python data_processor.py
```



* train

```cmd
python train_Lab.py --num_epochs 10 --device cuda:0
python train_rgb.py --num_epochs 10 --device cuda:0
```

* inference

```cnd
python test_Lab.py
python test_rgb.py
```

