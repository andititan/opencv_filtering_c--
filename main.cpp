#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){
    Mat image = imread("../27307090.jpg");
    if ( !image.data ){
        printf("No image data \n");
        return -1;
    }

    Mat frame_HSV;
    cvtColor(image, frame_HSV, COLOR_BGR2HSV);

    //lower boundary RED color range values; Hue (0 - 10)
    Mat lower_mask;
    inRange(frame_HSV, Scalar(0, 100, 20), Scalar(10, 255, 255), lower_mask);

    //upper boundary RED color range values; Hue (160 - 180)
    Mat upper_mask;
    inRange(frame_HSV, Scalar(160, 100, 20), Scalar(179,255,255), upper_mask);

    Mat full_mask = lower_mask + upper_mask;

    morphologyEx(full_mask, full_mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    morphologyEx(full_mask, full_mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));

    namedWindow("Original", WINDOW_AUTOSIZE );
    imshow("Original", image);

    namedWindow("New", WINDOW_AUTOSIZE );
    imshow("New", full_mask);

    waitKey(0);

    return 0;
}