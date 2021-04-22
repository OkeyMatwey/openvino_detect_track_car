#include <iostream>
#include <vector>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <ie_core.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace std;

auto net = cv::dnn::readNetFromModelOptimizer("../vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.xml","../vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin");
//auto net = cv::dnn::readNetFromCaffe("../SSD_512x512/deploy.prototxt", "../SSD_512x512/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel");
map<cv::Point, vector<cv::Point>> tracks;

float euclideanDist(const cv::Point& p, const cv::Point& q) {
    cv::Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void process(cv::Mat &img)
{
    cv::resize(img, img, cv::Size(672,384));
    auto blob = cv::dnn::blobFromImage(img,1);
    net.setInput(blob);
    cv::Mat out = net.forward().reshape(0,1);
    float image_id, label, conf, x_min, y_min, x_max, y_max;
    bool b = false;
    for(int c = 0; c < out.cols; c+=7)
    {
        image_id = out.at<float>(c+0);
        label = out.at<float>(c+1);
        conf = out.at<float>(c+2);
        x_min = out.at<float>(c+3)*672;
        y_min = out.at<float>(c+4)*384;
        x_max = out.at<float>(c+5)*672;
        y_max = out.at<float>(c+6)*384;
        if(x_max-x_min < 150 && conf > 0.4)
        {
            cv::rectangle(img, cv::Point(x_min,y_min), cv::Point(x_max,y_max),cv::Scalar(255,255,255),1);
            cv::Point new_centr(x_max-x_min, y_max-y_min);
            for(auto &old_centr : tracks)
            {
                if(euclideanDist(old_centr.first, new_centr) > 5)
                {
                    old_centr.second.push_back(new_centr);
                    b = true;
                    break;
                }

            }
            if(!b)
                tracks[new_centr] = vector<cv::Point>{new_centr};
            for(auto &old_centr : tracks)
            {
                const cv::Point *points = old_centr.second.data();
                const int *n = new int(5);
                cv::polylines(img, points, n, 2, cv::Scalar(255,255,255),1);
            }
        }
        if (label == -1)
            break;
    }
    cout << out.rows  << " " << out.cols << endl;
}

int main()
{
//    core.ReadNetwork("../vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin");
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

