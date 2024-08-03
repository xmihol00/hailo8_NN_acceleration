#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Flag to check if the mouse button is pressed
bool isDrawing = false;
// Starting point of the line
Point startPoint;

void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
    // Get the reference to the image
    Mat &image = *(Mat *)userdata;

    // When the left mouse button is pressed, start drawing
    if (event == EVENT_LBUTTONDOWN)
    {
        isDrawing = true;
        startPoint = Point(x, y);
    }
    // When the mouse is moved, draw a line from the starting point to the current mouse position
    else if (event == EVENT_MOUSEMOVE)
    {
        if (isDrawing)
        {
            Mat tempImage = image.clone();
            line(tempImage, startPoint, Point(x, y), Scalar(0, 255, 0), 2);
            imshow("Image", tempImage);
        }
    }
    // When the left mouse button is released, stop drawing
    else if (event == EVENT_LBUTTONUP)
    {
        isDrawing = false;
        line(image, startPoint, Point(x, y), Scalar(0, 255, 0), 2);
        imshow("Image", image);
    }
}

int main()
{
    // Load an image from file
    Mat image = imread("kitten.jpg");
    if (image.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Create a window
    namedWindow("Image", WINDOW_AUTOSIZE);

    // Set the mouse callback function to handle mouse events
    setMouseCallback("Image", mouseCallback, &image);

    // Display the image
    imshow("Image", image);

    // Wait until the user presses the 'q' key
    while (true)
    {
        char key = (char)waitKey(1);
        if (key == 'q')
        {
            break;
        }
    }

    // Clean up
    destroyAllWindows();
    return 0;
}
