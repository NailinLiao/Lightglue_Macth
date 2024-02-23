#include <iostream>
#include "LightglueMatch.h"


cv::Mat rotateImage(const cv::Mat &src, double angle, cv::Point2f center) {
    // 计算旋转矩阵
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

    // 获取旋转后图像的大小
    cv::Rect bbox = cv::RotatedRect(cv::Point2f(0, 0), src.size(), angle).boundingRect();
    cv::Mat dst(bbox.height, bbox.width, src.type());

    // 旋转图像
    cv::warpAffine(src, dst, rot_mat, dst.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    return dst;
}


// 函数定义：缩放图像并将其粘贴到另一张图像上
cv::Mat pasteScaledImage(cv::Mat src, cv::Mat dst) {
    // 缩小源图像
    cv::Mat scaledSrc;
    cv::resize(src, scaledSrc, cv::Size(), 0.3, 0.3); // 缩小为原尺寸的1/5

    // 计算目标位置
    int x = 0; // 左上角x坐标
    int y = 0; // 左上角y坐标

    // 粘贴到目标图像上
    cv::Rect roi(x, y, scaledSrc.cols, scaledSrc.rows);
    scaledSrc.copyTo(dst(roi));

    return dst;
}

int main() {
    cv::VideoCapture cap("../resource/test.mp4"); // 替换为你的视频文件路径
    cv::Mat Mat_satellite = cv::imread("../resource/ref.jpg");
    // 使用示例：
    double rotationAngle = 120.0; // 你想要旋转的角度
    cv::Mat rotatedImg = rotateImage(Mat_satellite, rotationAngle,
                                     cv::Point2f(Mat_satellite.cols / 2.0, Mat_satellite.rows / 2.0));


    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }


    // 输出视频的参数设置
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 创建VideoWriter对象以写入视频
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                           cv::Size(frameWidth, frameHeight));


    cv::Mat frame;
    cv::Mat old_frame;
    cap >> old_frame; // 读取第一帧
//    old_frame = rotatedImg.clone();
    cv::Point aim_point = cv::Point(960, 540);
    int radius = 5; // 圆的半径
    cv::Scalar color(0, 255, 255); // 蓝色圆圈 (BGR)
    int thickness = 2; // 线条粗细
    cv::circle(old_frame, aim_point,
               radius, color,
               thickness);

    LightglueMatch lightglueMatch("../resource/superpoint.rknn", "../resource/lightglue_3layers.rknn", 256);
    int i = 0;

    while (cap.read(frame)) { // 循环处理前500帧

        i++;

        if (i % 10 == 0) {
            int matchPtr_statu = lightglueMatch.get_statu();

            if (matchPtr_statu != 1) {
                std::cout << "Frame :" << i << " async STAY " << std::endl;
                lightglueMatch.async(old_frame, frame, aim_point, false);
            }

            matchPtr_statu = lightglueMatch.get_statu();

            while (matchPtr_statu == 1) {
//            std::cout << "Frame :" << i << " sleep_for  " << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                matchPtr_statu = lightglueMatch.get_statu();
            }

            if (matchPtr_statu == 2) {
                cv::Point2f aim_point = lightglueMatch.syncronize();
                std::cout << "Frame :" << i << " Map HIT" << std::endl;
                int radius = 5; // 圆的半径
                cv::Scalar color(0, 255, 255); // 蓝色圆圈 (BGR)
                int thickness = 2; // 线条粗细
                cv::circle(frame, aim_point,
                           radius, color,
                           thickness);
            } else if (matchPtr_statu == 0) {
                std::cout << "Frame :" << i << " Map NULL" << std::endl;
            }
            // 将原始图片缩小并贴到目标图片上
            cv::Mat targetImage = pasteScaledImage(old_frame, frame);

            // 保存结果图片
//        cv::imshow("frame", frame);
            cv::imshow("targetImage", targetImage);
            cv::imwrite(std::to_string(i) + "_targetImage.jpg", targetImage);

            cv::waitKey(1);
        }

    }
    // 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}
