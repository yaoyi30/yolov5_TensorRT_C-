#include "yolov5.h"
#include "common/common.hpp"
#include <chrono>
#include <algorithm>

YOLOv5::YOLOv5(const std::string &config_file,const std::string &label_file) {

    onnx_file = config_file;
    engine_file = "./yolov5s.trt";
    labels_file = label_file;
    BATCH_SIZE = 1;
    INPUT_CHANNEL = 3;
    IMAGE_WIDTH = 640;
    IMAGE_HEIGHT = 640;
    obj_threshold = 0.4;
    nms_threshold = 0.45;
    strides={8,16,32};
    num_anchors={3,3,3};
    anchors={{10,13},{16,30},{33,23},{30,61},{62,45},{59,119},{116,90},{156,198},{373,326}};
    assert(strides.size() == num_anchors.size());
    coco_labels = readCOCOLabel(labels_file);
    CATEGORY = coco_labels.size();
    int index = 0;
    for (const int &stride : strides)
    {
        grids.push_back({num_anchors[index], int(IMAGE_HEIGHT / stride), int(IMAGE_WIDTH / stride)});
    }
}

YOLOv5::~YOLOv5() = default;

void YOLOv5::Init_Model() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnxToTRTModel(onnx_file, engine_file, engine, BATCH_SIZE);
        assert(engine != nullptr);
    }
}

std::vector<YOLOv5::DetectRes> YOLOv5::Inference(cv::Mat &src_img) {

    std::vector<YOLOv5::DetectRes> result;

    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;

    result = EngineInference(src_img, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
    return result;
}

std::vector<YOLOv5::DetectRes> YOLOv5::EngineInference(cv::Mat &src_img, const int &outSize, void **buffers,
                             const std::vector<int64_t> &bufferSize, cudaStream_t stream) {


    std::vector<float>curInput = prepareImage(src_img);

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    std::cout << "host2device" << std::endl;
    cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    // do inference
    auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(BATCH_SIZE, buffers);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference take: " << total_inf << " ms." << std::endl;

    auto *out = new float[outSize * BATCH_SIZE];
    cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<YOLOv5::DetectRes> boxes = postProcess(src_img, out, outSize);
    
    return boxes;
}

std::vector<float> YOLOv5::prepareImage(cv::Mat &src_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();

    float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

    //HWC TO CHW
    int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * 2),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * 1),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * 0)
    };
    cv::split(flt_img, split_img);

    return result;
}

std::vector<YOLOv5::DetectRes> YOLOv5::postProcess(const cv::Mat &src_img, float *output,
                                                                const int &outSize) {
    std::vector<DetectRes> result;
    float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
    float *out = output;
    int position = 0;
    for (int n = 0; n < (int)grids.size(); n++)
    {
        for (int c = 0; c < grids[n][0]; c++)
        {
            std::vector<int> anchor = anchors[n * grids[n][0] + c];
            for (int h = 0; h < grids[n][1]; h++)
                for (int w = 0; w < grids[n][2]; w++)
                {
                    float *row = out + position * (CATEGORY + 5);
                    position++;
                    DetectRes box;
                    auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
                    box.prob = row[4] * row[max_pos - row];
                    if (box.prob < obj_threshold)
                        continue;
                    box.classes = coco_labels[max_pos - row - 5];
                    box.x = (row[0] * 2 - 0.5 + w) / grids[n][2] * IMAGE_WIDTH * ratio;
                    box.y = (row[1] * 2 - 0.5 + h) / grids[n][1] * IMAGE_HEIGHT * ratio;
                    box.w = pow(row[2] * 2, 2) * anchor[0] * ratio;
                    box.h = pow(row[3] * 2, 2) * anchor[1] * ratio;
                    result.push_back(box);
                }
        }
    }
    NmsDetect(result);

    return result;
}

void YOLOv5::NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}

float YOLOv5::IOUCalculate(const YOLOv5::DetectRes &det_a, const YOLOv5::DetectRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
