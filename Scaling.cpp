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


cv::Mat resizeWithPadding(const cv::Mat &src, float widthRatio, float heightRatio) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;

    // 检查是否需要调整尺寸
    if (widthRatio == 1.0f && heightRatio == 1.0f) {
        return src.clone();  // 如果比例为1，直接返回原图拷贝
    }

    // 计算目标宽度和高度
    int dstWidth = static_cast<int>(srcWidth * widthRatio);
    int dstHeight = static_cast<int>(srcHeight * heightRatio);

    // 处理放大情况
    if (dstWidth >= srcWidth && dstHeight >= srcHeight) {
        // 裁切中心区域并缩放至原始大小

        int cropX = (srcWidth - int(srcWidth / widthRatio)) / 2;
        int cropY = (srcHeight - int(srcHeight / widthRatio)) / 2;
        cv::Rect roi(cropX, cropY, int(srcWidth / widthRatio), int(srcHeight / widthRatio));
        cv::Mat croppedImage = src(roi).clone();
        cv::resize(croppedImage, croppedImage, cv::Size(srcWidth, srcHeight), 0, 0, cv::INTER_LINEAR);
        return croppedImage;
    }
        // 处理缩小情况
    else {
        cv::Mat resizedImage;
        cv::resize(src, resizedImage, cv::Size(dstWidth, dstHeight), 0, 0, cv::INTER_LINEAR);

        // 创建一个与原始尺寸相同的全黑背景图像
        cv::Mat paddedImage(cv::Size(srcWidth, srcHeight), src.type(), cv::Scalar(0, 0, 0));

        // 计算填充位置
        int padLeft = (srcWidth - dstWidth) / 2;
        int padRight = srcWidth - dstWidth - padLeft;
        int padTop = (srcHeight - dstHeight) / 2;
        int padBottom = srcHeight - dstHeight - padTop;

        // 将缩小后的图像粘贴到合适的位置
        cv::copyMakeBorder(resizedImage, paddedImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));

        return paddedImage;
    }
}

int main() {

//    读取卫星图，旋转卫星图，resie标准化卫星图；

    cv::Mat Mat_satellite = cv::imread("../resource/缩放卫星图实验/旺角/旺角2km.jpeg");

    cv::VideoCapture cap("../resource/缩放卫星图实验/旺角/旺角6km-0m.mp4"); // 替换为你的视频文件路径

    if (Mat_satellite.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1; // 或者执行其他错误处理操作
    }

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Video total_frames:" << total_frames << std::endl;
//    Mat_satellite = rotateImage(Mat_satellite, 180, cv::Point2f(Mat_satellite.cols / 2, Mat_satellite.rows / 2));
    cv::resize(Mat_satellite, Mat_satellite,
               cv::Size(1920, 1080));

//    读取视频
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

//    获取视频的帧总数；
//    帧号弹目距转换

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

    // 绘制全画面十字线
    int line_thickness = 2; // 线条粗细
    int start_distance = 6000;
    int satellite_distance = 2000;

    std::pair<cv::Mat, cv::Point2f> ret_corp;
    std::pair<cv::Mat, cv::Point2f> satellite_corp;


//    cv::Point2f aim_point_satellite = cv::Point(Mat_satellite.cols / 2, Mat_satellite.rows / 2);
//    cv::Point2f aim_point_frame;
//
//
//    cv::line(Mat_satellite, cv::Point(0, aim_point_satellite.y),
//             cv::Point(Mat_satellite.cols - 1, aim_point_satellite.y),
//             cv::Scalar(0, 0, 0), line_thickness);
//    // 绘制垂直线
//    cv::line(Mat_satellite, cv::Point(aim_point_satellite.x, 0),
//             cv::Point(aim_point_satellite.x, Mat_satellite.rows - 1),
//             cv::Scalar(0, 0, 0), line_thickness);
    bool using_pyramid = false;
    bool using_resize = true;
    int cout_hit = 0;
//    对比实验
    while (cap.read(frame)) {
//        缩放卫星图
        float now_distance = start_distance - i * (start_distance / total_frames);

        int matchPtr_statu = lightglueMatch.get_statu();
        while (matchPtr_statu == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            matchPtr_statu = lightglueMatch.get_statu();
        }
        std::cout << "当前弹目距离 :" << now_distance << std::endl;
        std::cout << "标准睁眼卫星图2km " << std::endl;

        if (matchPtr_statu == 2) {
            std::cout << "Frame :" << i << " HIT" << std::endl;
            cout_hit++;
            std::pair<cv::Mat, cv::Point> _ret = lightglueMatch.syncronize();
            cv::Point2f aim_point_ret = _ret.second;
            if (!using_resize) {
                cv::Point2f aim_point_frame;//=ret+corp_point
                cv::Point2f aim_point_Mat_satellite_wap;//=ret in satellite_wap
                aim_point_frame.y = aim_point_ret.y + ret_corp.second.y;
                aim_point_frame.x = aim_point_ret.x + ret_corp.second.x;

                cv::line(ret_corp.first, cv::Point(0, aim_point_ret.y),
                         cv::Point(ret_corp.first.cols - 1, aim_point_ret.y),
                         cv::Scalar(0, 0, 255), line_thickness);
                // 绘制垂直线
                cv::line(ret_corp.first, cv::Point(aim_point_ret.x, 0),
                         cv::Point(aim_point_ret.x, ret_corp.first.rows - 1),
                         cv::Scalar(0, 0, 255), line_thickness);

                cv::Point2f aim_point_satellite = cv::Point(satellite_corp.first.cols / 2,
                                                            satellite_corp.first.rows / 2);


                cv::line(satellite_corp.first, cv::Point(0, aim_point_satellite.y),
                         cv::Point(satellite_corp.first.cols - 1, aim_point_satellite.y),
                         cv::Scalar(0, 0, 255), line_thickness);
                // 绘制垂直线
                cv::line(satellite_corp.first, cv::Point(aim_point_satellite.x, 0),
                         cv::Point(aim_point_satellite.x, satellite_corp.first.rows - 1),
                         cv::Scalar(0, 0, 255), line_thickness);


                cv::Mat targetImage = pasteScaledImage(satellite_corp.first, ret_corp.first);//Mat_satellite_wap

                // 保存结果图片
                cv::imshow("show", targetImage);
                cv::moveWindow("show", 0, 0); // 左上角rm
                writer.write(targetImage);


            } else {
                cv::line(frame, cv::Point(0, aim_point_ret.y),
                         cv::Point(frame.cols - 1, aim_point_ret.y),
                         cv::Scalar(0, 0, 255), line_thickness);
                // 绘制垂直线
                cv::line(frame, cv::Point(aim_point_ret.x, 0),
                         cv::Point(aim_point_ret.x, frame.rows - 1),
                         cv::Scalar(0, 0, 255), line_thickness);

                cv::Point2f aim_point_satellite = cv::Point(Mat_satellite.cols / 2,
                                                            Mat_satellite.rows / 2);

                cv::line(Mat_satellite, cv::Point(0, aim_point_satellite.y),
                         cv::Point(Mat_satellite.cols - 1, aim_point_satellite.y),
                         cv::Scalar(0, 0, 255), line_thickness);
                // 绘制垂直线
                cv::line(Mat_satellite, cv::Point(aim_point_satellite.x, 0),
                         cv::Point(aim_point_satellite.x, Mat_satellite.rows - 1),
                         cv::Scalar(0, 0, 255), line_thickness);
                // 保存结果图片
                cv::Mat targetImage = pasteScaledImage(Mat_satellite, frame);//Mat_satellite_wap

                cv::imshow("show", targetImage);
                cv::moveWindow("show", 0, 0); // 左上角rm
                writer.write(targetImage);

            }
        } else {
            std::cout << "Frame :" << i << " LOSS" << std::endl;

        }

        if (using_pyramid) {
            if (now_distance > 5000) {
                Mat_satellite = cv::imread("../resource/缩放卫星图实验/旺角/旺角5km.jpeg");
                satellite_distance = 5000;
            } else if (now_distance > 3000) {
                Mat_satellite = cv::imread("../resource/缩放卫星图实验/旺角/旺角4km.jpeg");
                satellite_distance = 4000;
            } else if (now_distance > 2000) {
                Mat_satellite = cv::imread("../resource/缩放卫星图实验/旺角/旺角3km.jpeg");
                satellite_distance = 3000;
            } else if (now_distance > 1000) {
                Mat_satellite = cv::imread("../resource/缩放卫星图实验/旺角/旺角2km.jpeg");
                satellite_distance = 2000;
            }
            float scaling = satellite_distance / now_distance;
            std::cout << "Video now_distance:" << now_distance << std::endl;
            std::cout << "Video satellite_distance:" << satellite_distance << std::endl;
            std::cout << "Video scaling:" << scaling << std::endl;
            Mat_satellite = resizeWithPadding(Mat_satellite, scaling, scaling);

        }

        if (using_resize) {
            lightglueMatch.async(Mat_satellite, frame, cv::Point(Mat_satellite.cols / 2, Mat_satellite.rows / 2),
                                 false);
        } else {

            ret_corp = crop_and_pad_image_(frame, int(frame.cols / 2), int(frame.rows / 2), 960, 544);

            satellite_corp = crop_and_pad_image_(Mat_satellite, int(Mat_satellite.cols / 2),
                                                 int(Mat_satellite.rows / 2), 960, 544);

            lightglueMatch.async(satellite_corp.first, ret_corp.first, cv::Point(960 / 2, 544 / 2), false);

        }

        i++;
        cv::waitKey(1);

    }
    std::cout << "  是否使用金字塔图像：" << using_pyramid << "  是否使用粗匹配：" << using_resize << "  共计帧数: :" << i
              << "  命中帧数:"
              << cout_hit << std::endl;
// 释放资源
    cap.release();

    writer.release();

    cv::destroyAllWindows();

    return 0;
}
