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


std::vector<cv::Point2f> calculate_crop_points(int original_height,
                                               int original_width,
                                               int window_height,
                                               int window_width,
                                               int overlap_pixels) {
    std::vector<cv::Point2f> crop_points;

    // 计算水平方向上的居中裁切起点
    int x_offset = (original_width - window_width) / 2;

    // 计算垂直方向的滑动步长
    int step_y = window_height - overlap_pixels;

    for (int y = 0; y <= original_height; y += step_y) {
        cv::Point2f cp;
        cp.x = x_offset;
        cp.y = y;
        crop_points.push_back(cp);
    }

    return crop_points;
}

int main() {

//    读取卫星图，旋转卫星图，resie标准化卫星图；

    cv::Mat Mat_satellite = cv::imread("../resource/1500.jpeg");
    cv::VideoCapture cap("../resource/test_video.mp4"); // 替换为你的视频文件路径

    if (Mat_satellite.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1; // 或者执行其他错误处理操作
    }
//    Mat_satellite = rotateImage(Mat_satellite, 180, cv::Point2f(Mat_satellite.cols / 2, Mat_satellite.rows / 2));
    cv::resize(Mat_satellite, Mat_satellite,
               cv::Size(1920, 1080));

//    读取视频
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

    LightglueMatch lightglueMatch("../resource/superpoint.rknn", "../resource/lightglue_3layers.rknn", 1024);
    int i = 0;

//    创建ORB
    ORB_WORKER oRB_WORKER = ORB_WORKER(0.2, frameWidth, frameHeight);

//    首先要查找共视区域
//标记共视区域查找状态
//        计算画面搜索裁切点；
    int img_height = frameHeight;
    int img_width = frameWidth;


    // 绘制全画面十字线
    int line_thickness = 2; // 线条粗细


    std::vector<cv::Point2f> crop_points = {cv::Point2f(img_width / 2, 0), cv::Point2f(img_width / 2, 384),
                                            cv::Point2f(img_width / 2, 768), cv::Point2f(img_width / 2, 1152),
                                            cv::Point2f(img_width / 2, 1536)};
    for (const auto &cp: crop_points) {
        std::cout << " FrameWidth:" << frameWidth << "  FrameHeight" << frameHeight << " Crop Point: (" << cp.x
                  << ", " << cp.y << ")" << std::endl;
    }

    bool loss = true;

    std::pair<cv::Mat, cv::Point2f> ret_corp;
    std::pair<cv::Mat, cv::Point2f> satellite_corp;


    cv::Point2f aim_point_Mat_satellite = cv::Point(Mat_satellite.cols / 2, Mat_satellite.rows / 2);
    cv::Point2f aim_point_frame;//=ret+corp_point
    cv::Point2f aim_point_Mat_satellite_wap;//=ret in satellite_wap
    cv::Point aim_point_Mat_satellite_corp;
    int inference_w = 960;
    int inference_h = 544;


    cv::Mat Mat_satellite_wap = Mat_satellite.clone();

    while (cap.read(frame)) { // 循环处理前500帧
        i++;
        oRB_WORKER.main_run(frame);
        if (i % 5 == 0 && i > 180) {
//            if (loss) {
            if (true) {
                cv::Point2f corp_point;
                int corp_index = 2;
                //corp_index = (i+1) % crop_points.size();
                corp_point = crop_points[corp_index];
                ret_corp = crop_and_pad_image_(frame, int(corp_point.x), int(corp_point.y), 1536, 870);
                int matchPtr_statu = lightglueMatch.get_statu();
                if (matchPtr_statu != 1) {
                    lightglueMatch.async(Mat_satellite, ret_corp.first, aim_point_Mat_satellite, false);
                    oRB_WORKER.reset_H();
                }
            } else {
                loss = true;
//                return 1;
                //此处执行精匹配
            }

            int matchPtr_statu = lightglueMatch.get_statu();

            while (matchPtr_statu == 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                matchPtr_statu = lightglueMatch.get_statu();
            }
            if (matchPtr_statu == 2) {
                std::cout << "Frame :" << i << " HIT" << std::endl;
                std::pair<cv::Mat, cv::Point> _ret = lightglueMatch.syncronize();
                cv::Point2f aim_point_ret = _ret.second;
                //                加上原图的裁切偏移量
                aim_point_frame.y = aim_point_ret.y + ret_corp.second.y;
                aim_point_frame.x = aim_point_ret.x + ret_corp.second.x;

                if (loss) {
                    cv::warpPerspective(Mat_satellite, Mat_satellite_wap, _ret.first, Mat_satellite.size());
                    aim_point_Mat_satellite_wap = aim_point_ret;
                } else {

                }
                loss = false;
            } else {
//                loss
                loss = true;
            }
// 将原始图片缩小并贴到目标图片上
            cv::Mat Mat_satellite_wap_show = Mat_satellite_wap.clone();

            if (!loss) {
                // 绘制水平线
                cv::line(frame, cv::Point(0, aim_point_frame.y), cv::Point(frame.cols - 1, aim_point_frame.y),
                         cv::Scalar(0, 255, 255), line_thickness);

                // 绘制垂直线
                cv::line(frame, cv::Point(aim_point_frame.x, 0), cv::Point(aim_point_frame.x, frame.rows - 1),
                         cv::Scalar(0, 255, 255), line_thickness);


                cv::line(Mat_satellite_wap_show, cv::Point(0, aim_point_Mat_satellite_wap.y),
                         cv::Point(Mat_satellite_wap_show.cols - 1, aim_point_Mat_satellite_wap.y),
                         cv::Scalar(0, 255, 255), line_thickness);

                // 绘制垂直线
                cv::line(Mat_satellite_wap_show, cv::Point(aim_point_Mat_satellite_wap.x, 0),
                         cv::Point(aim_point_Mat_satellite_wap.x, Mat_satellite_wap_show.rows - 1),
                         cv::Scalar(0, 255, 255), line_thickness);
            }

            cv::Mat targetImage = pasteScaledImage(Mat_satellite_wap_show, frame);//Mat_satellite_wap

            // 保存结果图片
            cv::imshow("show", targetImage);
            cv::moveWindow("show", 0, 0); // 左上角rm

            cv::imwrite(std::to_string(i) + "_targetImage.jpg", targetImage);
            writer.write(targetImage);
            cv::waitKey(1);

        }
        cv::waitKey(1);

    }

// 释放资源
    cap.

            release();

    writer.

            release();

    cv::destroyAllWindows();

    return 0;
}
