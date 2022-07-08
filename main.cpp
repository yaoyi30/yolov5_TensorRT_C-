#include <opencv2/opencv.hpp>
#include "src/yolov5.h"

int main()
{
    std::vector<YOLOv5::DetectRes> result;
    std::string onnx_file = "/media/yao/Data/yolov5-tensorrt/models/yolov5s.onnx";
    std::string label_file = "/media/yao/Data/yolov5-tensorrt/src/coco.names";
    cv::Mat org_img = cv::imread("/media/yao/Data/yolov5-tensorrt/images/bus.jpg");
    YOLOv5 YOLOv5(onnx_file,label_file);
    YOLOv5.Init_Model();
    result = YOLOv5.Inference(org_img);
    for(const auto &rect : result)
    {
        std::string name = rect.classes;
        cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,255,0), 2);
        cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
        cv::rectangle(org_img, rst,cv::Scalar(255,255,0), 2, cv::LINE_8, 0);
    }
    cv::imwrite("1.jpg", org_img);
    return 0;
}
