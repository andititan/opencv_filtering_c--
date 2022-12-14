#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){
    Mat image = imread("../a3d2f76670f48030854736790e9dd1e4.jpg");
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

    // Set up SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;

    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByCircularity = false;
    params.filterByColor = true;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 255;
    params.blobColor = 255;

    // Filter by Area
    params.filterByArea = true;
    params.minArea = 0;
    params.maxArea = 90000000;
    params.minDistBetweenBlobs = 0.0f;

    // Perform the detection
    cv::Ptr<cv::SimpleBlobDetector> detector = 
    cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(full_mask, keypoints);

    cv::Mat im_with_keypoints;
    drawKeypoints(full_mask, keypoints, im_with_keypoints, cv::Scalar(0, 0, 255), 
    cv::DrawMatchesFlags::DEFAULT);

    namedWindow("Original", WINDOW_AUTOSIZE );
    imshow("Original", image);

    namedWindow("New", WINDOW_AUTOSIZE );
    imshow("New", im_with_keypoints);

    waitKey(0);

    return 0;
}