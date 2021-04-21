#include <iostream>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <ie_core.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;

Core core("../vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.xml");
CNNNetwork network = core.ReadNetwork("../vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin");
std::string imageInputName, imInfoInputName;

void process(cv::Mat &img)
{

}

int main()
{
    core.ReadNetwork("../vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin");
    cv::Mat img;
    cv::VideoCapture cap("d:/Road_traffic.mp4");
    if (!cap.isOpened()) {
        std::cout << "video file not open";
        return -1;
    }
    int start = clock();
    int end;
    int fps = 0;
    while (true)
    {
        cap.read(img);
        if (img.empty()) {
            std::cout << "frame grabbed error\n";
            continue;
        }
        process(img);
        imshow("video", img);
        if (cv::waitKey(5) >= 0)
            break;
        fps++;
        end = clock();
        if( (end - start) / CLOCKS_PER_SEC > 1)
        {
            std::cout << fps << std::endl;
            fps = 0;
            start = end;
        }
    }
    return 0;
}

