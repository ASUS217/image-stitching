#include <iostream>  
#include <stdio.h>  
#include "opencv2/core.hpp"  
#include "opencv2/core/utility.hpp"  
#include "opencv2/core/ocl.hpp"  
#include "opencv2/imgcodecs.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/features2d.hpp"  
#include "opencv2/calib3d.hpp"  
#include "opencv2/imgproc.hpp"  
#include"opencv2/flann.hpp"  
#include"opencv2/xfeatures2d.hpp"  
#include"opencv2/ml.hpp"
#include <opencv2/stitching.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

bool try_use_gpu = false; 
vector<Mat> imgs;   //定义一个图像容器imgs，用于存放源图像
string result_name = "dst1.jpg";
int main(int argc, char * argv[])
{
	Mat img1 = imread("F:\\picture\\拼接\\trees_000.jpg");
	Mat img2 = imread("F:\\picture\\拼接\\trees_001.jpg");

	imshow("p1", img1);
	imshow("p2", img2);

	if (img1.empty() || img2.empty())
	{
		cout << "Can't read image" << endl;
		return -1;
	}
	imgs.push_back(img1);    //把源图像img1放入图像容器imgs中
	imgs.push_back(img2);    //把源图像img2放入图像容器imgs中


	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	// 使用stitch函数进行拼接
	Mat pano;
	Stitcher::Status status = stitcher.stitch(imgs, pano);
	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}
	imwrite(result_name, pano);
	Mat pano2 = pano.clone();
	// 显示源图像，和结果图像
	imshow("全景图像", pano);
	if (waitKey() == 27)
		return 0;
}
