## Sharpen

Image sharpening algorithms are a technique used to enhance details and edges in images. These methods can all be used for image sharpening. In short, sharpening is about enhancing the difference on edges (what is an edge, see image edge detection, etc.) to highlight the color brightness value between pixels around the edge.

Edge detection is to find the edge information of the image, that is, the place where the pixel changes greatly, that is, to calculate the gradient value for the derivative, the commonly used method mainly involves the calculation of first-order differentiation and second-order differentiation. Specifically, you should choose the appropriate method according to your needs.

## Method

`Good detection results`: low false detection rate of edges, avoiding the detection of false edges while detecting true edges; The edge position of the marker should be as close as possible to the real edge position on the image.

When calculating the `gradient of the image` (the intensity of change between pixels), unlike the differential derivation of one-dimensional data, convolutional calculations are mainly used in two-dimensional or above-dimensional arrays (what is convolutional computing, self-search, recently known as due to the fire of deep learning networks), the following are several common image sharpening algorithms:
- 1.*[拉普拉斯算子](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E7%AE%97%E5%AD%90)*（Laplace operator, Laplacian）
- 2.*[高斯——拉普拉斯算子](https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm)*(Laplacian of Gaussian)

- 3.*[Roberts算子](https://zhuanlan.zhihu.com/p/506408180)*

- 4.*[Sobel算子](https://zh.wikipedia.org/wiki/%E7%B4%A2%E4%BC%AF%E7%AE%97%E5%AD%90)*
- 5.*[Prewitt算法](zhuanlan.zhihu.com/p/266180689)*

# 锐化综述

>图像锐化算法是一种用于增强图像中细节和边缘的技术。这些方法都可以用于图像锐化。 简言之，锐化就是增强边缘上的差异（何为[边缘检测](https://zh.wikipedia.org/wiki/%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B)，可查看图像边缘检测等内容），来突出边缘周围像素间颜色亮度值。

>边缘检测在于找到图像边缘信息，就是像素变化较大的地方，即为求导数算梯度值，常用的方法主要涉及的计算就是<u>一阶微分</u>与<u>二阶微分</u>等计算。具体应该根据需要选择适合的方法。

`好的检测结果`：对边缘的错误检测率低，在检测出真实边缘的同时，避免检测虚假边缘；标记的边缘位置要和图像上真正的边缘位置尽量接近。

在计算`图像的梯度`（像素间变化强度）时，与一维数据微分求导不同，在二维或以上维度数组主要用的是卷积计算（何为卷积计算，自行搜索，最近为大家所熟知是由于深度学习网络的大火），以下是常见的几种图像锐化算法：

## 1.	拉普拉斯算子锐化：
拉普拉斯是一种二阶微分计算。通过在原始图像上应用拉普拉斯滤波器来增强图像中的边缘和细节，边缘定位准确，对噪声非常敏感。
推导出的拉普拉斯卷积核两个常用示例如下。拉普拉斯卷积核中心值比较大，意味着在做计算中，图像当前像素值占权重较大，锐化时为了增大与周围像素的差异，周围像素值较小。在卷积核中，中心像素值为9，表示将中心像素的强度值增加，周围像素的强度值减少。这样，锐化后的图像将具有更强的边缘和更清晰的细节。这就会使得计算后突变的像素更加突出，慢变的像素相对弱化。

![Aaron Swartz](https://github.com/CSsaan/MyISharpen/raw/main/Img/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E7%AE%97%E5%AD%90.jpg)
<p style="text-align: center;">拉普拉斯算子</p>
 
![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/%E5%8E%9F%E5%9B%BE.jpg)
<p style="text-align: center;">原图</p>
 
![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/laplace.jpg)

<p style="text-align: center;">laplace</p>
<p style="text-align: center;">（不平滑，但效果明显）</p>


**缺点**：这种方法可能会导致图像噪声的增加，丢失部分边缘方向信息，导致一些不连续的检测边缘。优点：把x、y两个方向均考虑了，不依赖于方向，对阶跃型边缘检测很准。

## 2.	高斯——拉普拉斯算子锐化（LoG）：
该方法相较于拉普拉斯算法，是一种更好的边缘检测器。它把高斯模糊平滑器和拉普拉斯锐化结合起来，先用高斯核对图像进行平滑去噪，再用拉普拉斯核对图像进行边缘检测锐化。
该卷积核的中心权重很大，周围随着距离增加权重系数逐渐降低。示例两种高斯——拉普拉斯卷积核如下： 

<figure>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/%E9%AB%98%E6%96%AF%E2%80%94%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF(%E5%B7%A6).jpg" width = 30%/>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/%E9%AB%98%E6%96%AF%E2%80%94%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF(%E4%B8%AD).jpg" width = 20%/>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/%E9%AB%98%E6%96%AF%E2%80%94%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF(%E5%8F%B3).jpg" width = 19%/>
（右为Usharpe Masking法）
</figure>



![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/%E5%8E%9F%E5%9B%BE.jpg)
<p style="text-align: center;">原图</p>
 
![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/LoG%EF%BC%88%E4%B8%AD%E9%97%B4%E7%AE%97%E5%AD%90%EF%BC%89.jpg)
<p style="text-align: center;">LoG（中间算子）</p>
<p style="text-align: center;">（相较于Laplace效果更加平滑）</p>

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Usharpe%20Masking%EF%BC%88%E5%8F%B3%E7%AE%97%E5%AD%90%EF%BC%89.jpg)

<p style="text-align: center;">Usharpe Masking（右算子）</p>
<p style="text-align: center;">（相较于Laplace效果更加平滑）</p>

**优点**：主要是对于拉普拉斯算子的优化，将高斯平滑考虑其中，降低了拉普拉斯算子对于噪声抑制能力差的问题。**缺点**：但同时无法避免对于一些尖锐的高频特征，也会有一定的平滑，其中高斯卷积的方差参数会对边缘检测效果有一定影响。

## 3.	Roberts算子锐化：
是一种斜向偏差分的梯度计算方法，梯度的大小代表边缘的强度，梯度的方向与边缘的走向垂直。卷积核如下：

<figure>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/Roberts%E7%AE%97%E5%AD%90%EF%BC%88%E5%B7%A6%EF%BC%89.jpg" width = 30%/>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/Roberts%E7%AE%97%E5%AD%90%EF%BC%88%E5%8F%B3%EF%BC%89.jpg" width = 30%/>
</figure>
<p style="text-align: center;">Roberts算子</p>
   
![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/%E5%8E%9F%E5%9B%BE.jpg)
<p style="text-align: center;">原图</p>

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Roberts%E6%95%88%E6%9E%9C%EF%BC%88%E5%B7%A6%E7%AE%97%E5%AD%90%EF%BC%89.jpg)
<p style="text-align: center;">Roberts（左算子）</p>

 ![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Roberts%E6%95%88%E6%9E%9C%EF%BC%88%E5%8F%B3%E7%AE%97%E5%AD%90%EF%BC%89.jpg)
<p style="text-align: center;">Roberts（右算子）</p>

**缺点**：对角线相减，定位比较高，但是没有平滑处理，直接减，容易丢失一部分边缘，不具备抗噪声的能力。**优点**：但对陡峭边缘且含噪声少的图像效果较好。

## 4.	Sobel算子锐化：
通过在原始图像上应用Sobel算子，可以增强图像中的水平和垂直边缘。这种方法比拉普拉斯算子锐化更平滑。卷积核如下：

<figure>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/Sobel%E7%AE%97%E5%AD%90%EF%BC%88%E5%B7%A6%E5%8F%B3%EF%BC%89.jpg" width = 50%/>
</figure>
<p style="text-align: center;">Sobel算子</p>
 
![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/%E5%8E%9F%E5%9B%BE.jpg)

<p style="text-align: center;">原图</p>

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Sobel%EF%BC%88%E5%B7%A6%E7%AE%97%E5%AD%90%EF%BC%89.jpg)

<p style="text-align: center;">Sobel（左算子）</p>

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Sobel%EF%BC%88%E5%8F%B3%E7%AE%97%E5%AD%90%EF%BC%89.jpg)

<p style="text-align: center;">Sobel（右算子）</p>

**缺点**：对噪声有一定平滑抑制的能力，但是还会存在虚假边缘检测，容易检测出多像素的边缘宽度。

## 5.	Prewitt算法：
Prewitt算法和sobel算法类似，只是卷积核的权重值不同。Prewitt算法的卷积核模板对于周围像素的中间位置像素权重与两侧权重相同，这使得周围3临近的像素对于当前像素的影响相同。卷积核如下：

<figure>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/Prewitt%EF%BC%88%E5%B7%A6%E5%8F%B3%EF%BC%89.jpg" width = 50%/>
</figure>
<p style="text-align: center;">Prewitt算子</p>

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/%E5%8E%9F%E5%9B%BE.jpg)

<p style="text-align: center;">原图</p>

 ![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Prewitt%EF%BC%88%E5%B7%A6%E7%AE%97%E5%AD%90%EF%BC%89.jpg)

<p style="text-align: center;">Prewitt（左算子）</p>

 ![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Prewitt%EF%BC%88%E7%AE%97%E5%AD%90%EF%BC%89.jpg)

<p style="text-align: center;">Prewitt（算子）</p>

## 6.	Canny算子锐化：
Canny算子锐化是一种多阶段边缘检测算法，可在保留更多真实边缘的同时减少噪声。由于Canny输出为二值数据（边缘、非边缘），锐化暂时未考虑该方法。
### Canny边缘检测算法主要包括以下五个步骤:
- 对图像进行高斯滤波，以去除噪声。
- 计算图像的梯度幅值和方向。
- 对梯度幅值进行非极大值抑制。
- 使用双阈值法进行边缘跟踪。
- 去除噪声并连接边缘。
- 
具体GLSL实现Canny边缘检测实现请看：链接（候补）。

## 7.	Unsharp Masking锐化： 
Unsharp Masking锐化也是一种常用的图像增强技术，它通过对原始图像进行高斯模糊并将其减去原始图像，然后对结果应用增强锐化滤波器，以增强图像的细节和边缘。
最初应用于暗房摄影。它的名字来源，即该技术使用模糊的或“不清晰的”负片图像来创建蒙版原始图像的。然后将反锐化蒙版与原始正像结合，创建一个比原始图像更不模糊的图像。生成的图像虽然更清晰，但对图像主体的表示可能不太准确。
卷积核为：<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/Unsharp%20Masking-1.jpg" width = 15%/>。其计算过程如下：

<figure>
<img src="https://github.com/CSsaan/MyISharpen/raw/main/Img/Unsharp%20Masking-2.jpg" width = 95%/></figure>

属于高斯——拉普拉斯算子的一种，其效果如下：

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/%E5%8E%9F%E5%9B%BE.jpg)

<p style="text-align: center;">原图</p>

![Aaron Swartz](https://github.com/CSsaan/MyImage/raw/main/Img/Unsharp%20Masking%EF%BC%88LoG%E7%AE%97%E5%AD%90%EF%BC%89.jpg)

<p style="text-align: center;">Unsharp Masking（LoG算子）</p>

## 代码实现例程：
主要通过 <u>*C++*</u>、<u>*Python*</u>、<u>*shader GLSL*</u>实现。
### (1). OpenCV（C++）：
```` C++
#include <opencv2/opencv.hpp>
using namespace cv;

int main() {
// 读取图像
Mat img = imread("image.jpg");
// 定义锐化卷积核(以拉普拉斯算子为例)
Mat kernel = (Mat_<float>(3,3) << -1,-1,-1,
                                   -1, 9,-1,
                                   -1,-1,-1);

// 应用锐化卷积核
Mat sharp_img;
filter2D(img, sharp_img, -1, kernel);

// 显示原图和锐化后的图像
namedWindow("Original Image");
namedWindow("Sharpened Image");
imshow("Original Image", img);
imshow("Sharpened Image", sharp_img);
waitKey(0);
destroyAllWindows();

return 0;
}
````

### (2). OpenCV（Python）：
```` python
import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg')

# 定义锐化卷积核(以拉普拉斯算子为例)
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

# 应用锐化卷积核
sharp_img = cv2.filter2D(img, -1, kernel)

# 显示原图和锐化后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Sharpened Image', sharp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
````

### (3). GLSL版本（片段着色器）：
```` GLSL
varying vec2 vv2_Texcoord;   // 纹理坐标
uniform sampler2D m_texture; // 原图片

vec3 mtexSample(const float x, const float y)
{
    vec2 uv = vv2_Texcoord + vec2(x / 1280.0, y / 720.0); // 纹理分辨率：1280,720
    lowp vec3 textureColor = texture2D(m_texture, uv);    // 图像纹理采样
    return textureColor;
}

vec3 sharpen(vec2 fragCoord, float strength)
{
    //卷积核 (以拉普拉斯算子为例)
    vec3 f =
        mtexSample(-1.0, -1.0) * -1.0 +
        mtexSample(0.0, -1.0) * -1.0 +
        mtexSample(1.0, -1.0) * -1.0 +

        mtexSample(-1.0, 0.0) * -1.0 +
        mtexSample(0.0, 0.0) * 9.0 +
        mtexSample(1.0, 0.0) * -1.0 +

        mtexSample(-1.0, 1.0) * -1.0 +
        mtexSample(0.0, 1.0) * -1.0 +
        mtexSample(1.0, 1.0) * -1.0;

    return mix(vec4(mtexSample(0.0, 0.0), 1.0), vec4(f, 1.0), strength).rgb;
}

void main()
{
    vec3 sharpened = sharpen(vv2_Texcoord, 1.0);
    gl_FragColor = vec4(sharpened, 1.0);
}
````
