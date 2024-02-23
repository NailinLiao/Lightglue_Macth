#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <deque>
#include <memory>
#include <atomic>
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <thread>
#include <pthread.h>
#include <random>
#include <ctime>
#include "rknn_api.h"

#define MATCH_DEBUG

class RKNNModel_ {
public:
    RKNNModel_(std::string path);

private:
    int model_len;
    unsigned char *model = nullptr;
    rknn_context ctx = 0;
    rknn_input_output_num io_num;

public:
    int set_input(rknn_input *inputs);

    void get_output(rknn_output *outputs);

    void release_output(rknn_output *outputs);

    int run(rknn_core_mask mask = RKNN_NPU_CORE_0_1_2);

    int get_num_output();

};


class SuperPoint {
public:
    SuperPoint(int _width, int _height, std::string path);

private:
    std::shared_ptr<RKNNModel_> model;
    int width, height;
    float *maxpool, *softmax_sum, *l2_sum;

    std::vector<cv::KeyPoint> nms(float *pool_t, int window_size = 9, float threshold = 0.00005, int borders = 4);

public:
    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect(cv::Mat img);
};


class LightglueMatch {
public:
    LightglueMatch(std::string model_superpoint_path, std::string model_lightglue_path, int point);

private:
    std::string model_path_lightglue, model_path_superpoint;
    std::shared_ptr<SuperPoint> model_superpoint;
    std::shared_ptr<RKNNModel_> model_lightglue;
    int point_num, RUN_number,loss_number;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Point2f aim_point, aim_point_detect;

    std::shared_ptr<std::thread> work_loop_thread_ptr;
    cv::Mat _update_frame, _satellite;
    std::atomic_bool ONLY_SP;
    std::atomic_int system_state;//  0 static,1 run;
    std::atomic_int HIT_state;//  0 static,1 run;
    std::atomic_int _async_flag; //  0,1,2 LIKE none, run, hit,
    int inference_w;
    int inference_h;



public:

    std::pair<cv::Mat, bool>
    lightGlue_inference(cv::Mat img0, cv::Mat img1, cv::Point2f img0_cut_point = cv::Point2f(0, 0),
                        cv::Point2f img1_cut_point = cv::Point2f(0, 0));

    std::pair<cv::Mat, bool> superPoint_inference(cv::Mat img0, cv::Mat img1);

    void async(cv::Mat frame, cv::Mat satellite, cv::Point aim_point, bool ONLY_SP = false);

    cv::Point syncronize(); //   return cv::Mat H = findHomography(pts0, pts1, cv::RANSAC);

    int get_statu();  //   return 0,1,2 LIKE none, run, end,
    void work_loop();  //   while 1 run in run_flag==true


};