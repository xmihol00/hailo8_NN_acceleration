#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("kitten.jpg");

    if (image.empty())
    {
        cerr << "Could not open or find the image" << endl;
        return -1;
    }

    string windowName = "Kitten";
    namedWindow(windowName);

    imshow(windowName, image);
    waitKey(0);
    destroyWindow(windowName);

    return 0;
}
