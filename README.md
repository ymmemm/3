# 实验三：图像关键点检测及图像拼接

#### 202410315058-姚钧耀

## 文件说明

本实验的代码在 `sample.ipynb` 文件中。

本实验的输入图像为 `a.jpg` 和 `b.jpg`。

## 一、实验目的

图像关键点检测及图像拼接实验报告，基于SIFT特征检测算法和FLANN特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接，掌握图像配准和融合的基本原理与实现方法。

## 二、实验内容与代码

### 1. 读取图像并预处理

读取两张图像作为拼接的原素材，再将图像转换为灰度图，以减少计算量并提高对光照和颜色变化的鲁棒性。

```python
import cv2
import numpy as np

img_a = cv2.imread("a.jpg")
img_b = cv2.imread("b.jpg")

gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
2. 关键点检测与特征匹配
使用SIFT算法检测特征点，并利用FLANN算法快速找到两幅图像特征点之间的对应关系，通过Lowe's比率测试过滤不可靠的匹配点。

python
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)
kp_b, des_b = sift.detectAndCompute(gray_b, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
3. 图像拼接与几何变换
利用RANSAC算法估计单应性矩阵，计算图像变换后的边界尺寸，最后通过透视变换和平移矩阵实现两幅图像的无缝空间对齐和内容融合。

python
src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

pts_a = np.float32([[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts_a, H)

all_corners = np.concatenate([dst_corners, np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

stitched_img = cv2.warpPerspective(img_a, translation_matrix @ H, (x_max - x_min, y_max - y_min))
stitched_img[-y_min : h_b - y_min, -x_min : w_b - x_min] = img_b

cv2.namedWindow('Stitched Result', cv2.WINDOW_NORMAL)
cv2.imshow('Stitched Result', stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## 三、实验原图展示
<img src="a.jpg" width="45%"> <img src="b.jpg" width="45%">
## 四、实验结果与分析
特征检测与匹配效果：
实验成功提取了图像的特征点并完成了匹配。由于图像纹理丰富，SIFT检测到了大量特征点，经过Lowe's比率测试后，保留下来的优质匹配点能够满足单应性矩阵的计算需求。

图像拼接质量评估：
单应性矩阵估计准确，拼接后的图像在重叠区域实现了较好的几何对齐，建筑物和树木的纹理基本吻合，没有出现明显的断裂或错位现象。但也存在因为透视变换产生的轻微黑边现象，这是正常的几何变换结果。

## 五、实验总结与心得
技术收获：
通过本次实验，我深入理解了SIFT特征检测算法的工作原理及其对尺度、旋转的鲁棒性。掌握了FLANN匹配器的使用以及Lowe's比率测试在提高匹配可靠性方面的作用。同时，也明白了单应性矩阵的估计是图像拼接的核心环节。

实践体会：
参数调优对最终结果影响很大。比如RANSAC的重投影误差阈值，设置不当会直接影响最终拼接的质量。这次实验让我将线性代数中的矩阵运算与实际视觉任务结合起来，不仅提高了代码编写能力，也加深了对计算机视觉底层原理的理解。
