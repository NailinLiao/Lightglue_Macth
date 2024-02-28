//
// Created by 廖乃琳 on 2024/2/21.
//
#include "ORB_WORKER.h"

ORB_WORKER::ORB_WORKER(float _speedup_rate, int _width, int _height) {
    speedup_rate = _speedup_rate;
    orb = cv::ORB::create();
    accumulate_H = cv::Mat::eye(3, 3, CV_64F);
    isFirstFrame = true;
    w = _width;
    h = _height;

    int new_width = cvRound(_width * speedup_rate);
    int new_height = cvRound(_height * speedup_rate);
// 计算裁切区域的左上角坐标，这里假设从中心裁切以保持图像居中
    int x = (_width - new_width) / 2;
    int y = (_height - new_height) / 2;
    ROI = cv::Rect(x, y, new_width, new_height);

#ifdef RGA
    rect.x = ROI.x;
    rect.y = ROI.y;
    rect.width = ROI.width;
    rect.height = ROI.height;
    tv_crop = new char[1920 * 1080 * 3];
    rkRga.RkRgaInit();
#endif

}

cv::Point2f ORB_WORKER::get_ORB_swap(cv::Point2f aim_point) {
    std::vector<cv::Point2f> transformed_points;
    std::vector<cv::Point2f> input_point;
//    cv::Point2f point_to_transform((aim_point.x - ROI.x) / speedup_rate, (aim_point.y - ROI.y) / speedup_rate);
    cv::Point2f point_to_transform(aim_point.x, aim_point.y);
    input_point.clear();
    input_point.push_back(point_to_transform);
    cv::perspectiveTransform(input_point, transformed_points, accumulate_H);
    cv::Point2f transformed_point = transformed_points[0];
    return transformed_point;
}

void ORB_WORKER::reset_H() {
    accumulate_H = cv::Mat::eye(3, 3, CV_64F);
    isFirstFrame = true;
}

cv::Mat ORB_WORKER::get_H() {
    return accumulate_H.clone();
}

void ORB_WORKER::forwad(cv::Mat _currFrame) {
    if (isFirstFrame) {
        orb->detectAndCompute(_currFrame, cv::noArray(), prevKeypoints, prevDescriptors);
        // 同样的操作应用于 currKeypoints
        for (cv::KeyPoint &kp: prevKeypoints) {
            kp.pt.x += ROI.x;
            kp.pt.y += ROI.y;
        }
        isFirstFrame = false;
    }

    orb->detectAndCompute(_currFrame, cv::noArray(), currKeypoints, currDescriptors);

    for (cv::KeyPoint &kp: currKeypoints) {
        kp.pt.x += ROI.x;
        kp.pt.y += ROI.y;
    }

    if (currDescriptors.rows < 30) {
        accumulate_H = cv::Mat::eye(3, 3, CV_64F);
        isFirstFrame = true;
        std::cout << " currDescriptors 提取 异常 currDescriptors.rows： " << currDescriptors.rows << std::endl;
    } else {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);

        std::vector<cv::DMatch> matches;

        matcher.match(prevDescriptors, currDescriptors, matches);
        // 应用比率测试或其他阈值筛选出稳定匹配

        // 对匹配结果按距离排序

        std::sort(matches.begin(), matches.end(),
                  [](const cv::DMatch &m1, const cv::DMatch &m2) {
                      return m1.distance < m2.distance;
                  });

        int nMatchesToKeep = std::min(100, static_cast<int>(matches.size()));
        std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + nMatchesToKeep);


        // 获取对应的关键点坐标
        std::vector<cv::Point2f> src_points, dst_points;
        for (const auto &match: good_matches) {
            src_points.push_back(prevKeypoints[match.queryIdx].pt);
            dst_points.push_back(currKeypoints[match.trainIdx].pt);
        }

        // 计算透视变换的单应矩阵
        accumulate_H *= cv::findHomography(src_points, dst_points, cv::RANSAC);
        // 更新前一帧为当前帧

        prevFrame = _currFrame.clone();

        prevKeypoints.swap(currKeypoints);
        prevDescriptors = currDescriptors.clone();
    }


}

cv::Mat ORB_WORKER::main_run(cv::Mat Frame) {

// 创建Rect对象
#ifdef RGA

    // 创建一个Mat对象来存储裁剪后的RGB图像

    // 将frame_crop_data复制到Mat中
        auto frame_data = wrapbuffer_virtualaddr(
                Frame.data, w, h, RK_FORMAT_RGB_888);

        auto frame_crop_data = wrapbuffer_virtualaddr(
                tv_crop, ROI.width, ROI.height, RK_FORMAT_RGB_888);
        int STATUS = imcrop(frame_data, frame_crop_data, rect);

        cv::Mat croppedImage(ROI.height, ROI.width, CV_8UC3, tv_crop);

        forwad(croppedImage);

#else
    cv::Mat cropped_Frame = Frame(ROI);
    forwad(cropped_Frame);
#endif

    return accumulate_H;
}