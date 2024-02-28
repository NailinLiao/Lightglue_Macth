//
// Created by 廖乃琳 on 2024/2/21.
//

#ifndef ORB_JITTER_ORB_WORKER_H
#define ORB_JITTER_ORB_WORKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

//#define RGA //如果是瑞芯微平台请尝试打开该鸡血开关

#ifdef RGA
#include "RgaUtils.h"
#include "RockchipRga.h"
#include "rga.h"
#include <im2d.h>
#include <im2d_type.h>

#endif

class ORB_WORKER {
private:
    float speedup_rate;
    cv::Ptr<cv::ORB> orb;
    cv::Mat prevFrame, currFrame;
    std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
    cv::Mat prevDescriptors, currDescriptors;
    cv::Rect ROI;
    cv::Mat accumulate_H;
    int w, h;
    char *tv_crop;

#ifdef RGA
    im_rect rect;
    RockchipRga rkRga;
#endif

    bool isFirstFrame;

    void forwad(cv::Mat currFrame);


public:
    ORB_WORKER(float _speedup_rate, int w, int h);

    cv::Mat main_run(cv::Mat Frame);

    cv::Mat get_H();

    void reset_H();


    cv::Point2f get_ORB_swap(cv::Point2f aim_point);
};

#endif //ORB_JITTER_ORB_WORKER_H
