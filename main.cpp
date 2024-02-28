#include <iostream>
#include "LightglueMatch.h"
#include "ORB_WORKER.h"


double calculateDistance(cv::Point2f p1, cv::Point2f p2) {
    return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

std::pair<cv::Mat, cv::Point2f>
crop_and_pad_image_(const cv::Mat &image, int center_x, int center_y, int width, int height) {
    // 获取图片高度和宽度
    int image_height = image.rows;
    int image_width = image.cols;

    // 计算裁剪区域
    int x1 = std::max(0, static_cast<int>(center_x - width / 2.0));
    int y1 = std::max(0, static_cast<int>(center_y - height / 2.0));
    int x2 = std::min(image_width, x1 + width);
    int y2 = std::min(image_height, y1 + height);

    // 创建全黑背景
    cv::Mat black_image(height, width, CV_8UC3, cv::Scalar(0));

    // 计算需要从原图中拷贝到黑背景中的区域
    int copy_x1 = std::max(x1, 0);
    int copy_y1 = std::max(y1, 0);
    int copy_x2 = std::min(x2, image_width);
    int copy_y2 = std::min(y2, image_height);

    // 计算在黑背景中的对应位置
    int black_x1 = copy_x1 - x1;
    int black_y1 = copy_y1 - y1;
    int black_x2 = black_x1 + (copy_x2 - copy_x1);
    int black_y2 = black_y1 + (copy_y2 - copy_y1);

    // 如果有需要拷贝的部分，执行拷贝操作
    if (copy_x1 < copy_x2 && copy_y1 < copy_y2) {
        cv::Rect roi(black_x1, black_y1, black_x2 - black_x1, black_y2 - black_y1);
        cv::Rect src_roi(copy_x1, copy_y1, copy_x2 - copy_x1, copy_y2 - copy_y1);
        image(src_roi).copyTo(black_image(roi));
    }
    // 返回裁剪并填充后的图像以及拷贝的起始坐标
    return std::make_pair(black_image, cv::Point2f(copy_x1, copy_y1));
}


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
    cv::resize(src, scaledSrc, cv::Size(), 0.3, 0.3); // 缩小

    // 计算目标位置
    int x = 0; // 左上角x坐标
    int y = 0; // 左上角y坐标

    // 粘贴到目标图像上
    cv::Rect roi(x, y, scaledSrc.cols, scaledSrc.rows);
    scaledSrc.copyTo(dst(roi));

    return dst;
}

int main() {
    cv::VideoCapture cap("../resource/1.mp4"); // 替换为你的视频文件路径
    cv::Mat Mat_satellite = cv::imread("../resource/11111.jpg");

    cv::resize(Mat_satellite, Mat_satellite,
               cv::Size(1920, 1080));

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
//    cap >> old_frame; // 读取第一帧
    old_frame = Mat_satellite.clone();


    cv::Point2f aim_point = cv::Point(960, 544);
//    cv::Point aim_point = cv::Point(960, 540);
    cv::Point2f points[4] = {cv::Point2f(0, 0), cv::Point2f(1920, 0), cv::Point2f(1920, 1080),
                             cv::Point2f(0, 1080),}; // 初始化四个二维点

    int radius = 5; // 圆的半径
    cv::Scalar color(0, 255, 255); // 蓝色圆圈 (BGR)
    int thickness = 2; // 线条粗细
    cv::circle(old_frame, aim_point,
               radius, color,
               thickness);

    LightglueMatch lightglueMatch("../resource/superpoint.rknn", "../resource/lightglue_3layers.rknn", 256);
    int i = 0;
    ORB_WORKER oRB_WORKER = ORB_WORKER(0.2, 1920, 1080);

    while (cap.read(frame) && i < 300) { // 循环处理前500帧

        i++;
//        在此处累积ORB
        oRB_WORKER.main_run(frame);
        std::cout << "oRB_WORKER.main_run" << std::endl;
        if (i % 5 == 0) {

            int matchPtr_statu = lightglueMatch.get_statu();

            if (matchPtr_statu != 1) {
                std::cout << "Frame :" << i << " async STAY " << std::endl;
                lightglueMatch.async(old_frame, frame, aim_point, false);
            }

            matchPtr_statu = lightglueMatch.get_statu();

            while (matchPtr_statu == 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                matchPtr_statu = lightglueMatch.get_statu();
            }

            if (matchPtr_statu == 2) {

                std::pair<cv::Mat, cv::Point> _ret = lightglueMatch.syncronize();
                //                  获取lightglueMatcher矩阵H
                //                  应用END_H 矩阵到卫星图

                bool H_hit = true;
                for (int i = 0; i < 4; ++i) {

                    std::vector<cv::Point2f> _srcPoints(1, points[i]);
                    std::vector<cv::Point2f> _dstPoints;
                    cv::Point2f new_point;
                    cv::perspectiveTransform(_srcPoints, _dstPoints, _ret.first);//转换到 弹 的视角
                    new_point.x = _dstPoints[0].x;
                    new_point.y = _dstPoints[0].y;
                    double distAB = calculateDistance(points[i], new_point);
                    std::cout << "points[i] distAB: " << distAB << std::endl;


                    if (distAB > 500) {
                        H_hit = false;
                    }
                }

                double det = cv::determinant(_ret.first);
                cv::SVD svd(_ret.first);
                double maxSingularValue = *std::max_element(svd.w.begin<double>(), svd.w.end<double>());
                double minSingularValue = *std::min_element(svd.w.begin<double>(), svd.w.end<double>());
                double conditionNumber = maxSingularValue / minSingularValue;
                std::cout << "Determinant: " << det << std::endl;
                std::cout << "Condition Number: " << conditionNumber << std::endl;

                if (H_hit) {
                    std::cout << "H_hit " << std::endl;

                    cv::Mat imgWarped;
                    cv::warpPerspective(old_frame, imgWarped, _ret.first, old_frame.size());
//                    添加ORB逻辑
                    aim_point = _ret.second;

                    cv::Mat ORB_H = oRB_WORKER.get_H();
                    cv::Mat imgWarped_ORB;

                    cv::warpPerspective(imgWarped, imgWarped_ORB, ORB_H, old_frame.size());
                    std::cout << "oRB_WORKER.warpPerspective" << std::endl;

                    old_frame = imgWarped_ORB.clone();

                    aim_point = oRB_WORKER.get_ORB_swap(aim_point);


                    oRB_WORKER.reset_H();
                    std::cout << "oRB_WORKER.reset_H" << std::endl;
                    std::cout << "Frame :" << i << " Map HIT" << std::endl;
                    cv::Scalar color(0, 255, 255); // 蓝色圆圈 (BGR)

                    // 绘制全画面十字线
                    int line_thickness = 2; // 线条粗细

                    // 绘制水平线
                    cv::line(frame, cv::Point(0, aim_point.y), cv::Point(frame.cols - 1, aim_point.y),
                             cv::Scalar(0, 255, 255), line_thickness);

                    // 绘制垂直线
                    cv::line(frame, cv::Point(aim_point.x, 0), cv::Point(aim_point.x, frame.rows - 1),
                             cv::Scalar(0, 255, 255), line_thickness);
                } else {
                    std::cout << " NO H_hit " << std::endl;
                }


            } else if (matchPtr_statu == 0) {
                std::cout << "Frame :" << i << " Map NULL" << std::endl;
            }
            // 将原始图片缩小并贴到目标图片上
            cv::Mat targetImage = pasteScaledImage(old_frame, frame);

            // 保存结果图片
            cv::imshow("targetImage", targetImage);
            cv::imwrite(std::to_string(i) + "_targetImage.jpg", targetImage);
            writer.write(targetImage);
            cv::waitKey(1);
        }

    }
    // 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}
