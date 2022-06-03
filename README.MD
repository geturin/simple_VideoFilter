# Simple Video Filter

**some video filters with AI**

## complete：
### 1.segmentation filter(CPU)
use [**mediapipe**](https://google.github.io/mediapipe/) on CPU<br>
[**[colab]**](https://colab.research.google.com/drive/1SE2JdqkBDvYHc6XcYEpjwfYNxctR6EkO?usp=sharing)<br>

![背景高斯模糊](images/samples/02.png)
![人物马赛克](images/samples/03.png)
### 2.depth filter(GPU)
use **Monocular Depth Estimation** model [**MiDaS**](https://github.com/isl-org/MiDaS) on GPU<br>
[**[colab]**](https://colab.research.google.com/drive/1_85OV4gQRv8FFykLKREi-FwusOO0t7z7?usp=sharing)<br>
![背景高斯模糊](images/samples/04.png)
![人物马赛克](images/samples/05.png)

## plan:
. segmentation filter(GPU)<br>
. stream to<br>
. sends frames to a virtual camera
