# PhotoMosaic
In the field of photographic imaging, a **photographic mosaic**, also known under the term **Photomosaic**, is a picture (usually a photograph) that has been divided into (usually equal-sized) tiled sections, each of which is replaced with another photograph that matches the target photo. When viewed at low magnifications, the individual pixels appear as the primary image, while close examination reveals that the image is in fact made up of many hundreds or thousands of smaller images.[1]

This project implements a program to create a Photomosaic version of a target image by using patching images. To form a high-quality Photomosaic image, one needs a certain amount of patching images. Here I use the Monet Paintings Dataset from the magnificent work of [CycleGAN](https://junyanz.github.io/CycleGAN/). I included the data used to generate the demo images in this repository, to find more data, please visit my [CycleGAN dataset](https://www.kaggle.com/datasets/sarihust/cyclegandata) on Kaggle.

### Requirement

```
python==3.9.7
numpy==1.21.5
opencv-python==4.6.0
```

The requirement above describes my environment configuration and are by no means the only possible configuration.

### Launch

To launch the program and generate a Photomosaic image of your own, you can choose your own patching image dataset and target image by changing the file path in `photomosaic.py` and then use `python` to run the script.

```shell
python photomosaic.py
```

Or you can import the module and complete the settings in the terminal.

```python
>>> from photomosaic import PhotoMosaic
>>> pm = PhotoMosaic(20)
>>> pm.load_patching_images('your patching dataset path')
>>> pm.load_target('your target path')
>>> pm.process()
>>> pm.store_result('your result path')
```

### References

[1] https://en.wikipedia.org/wiki/Photographic_mosaic

### Contact

Feel free to contact me by mailing [Hanhui Wang](mailto:1727429088@qq.com) if you have any questions.
