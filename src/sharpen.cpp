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
