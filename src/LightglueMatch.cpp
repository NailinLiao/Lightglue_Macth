#include "LightglueMatch.h"

std::pair<cv::Mat, cv::Point2f>
crop_and_pad_image(const cv::Mat &image, int center_x, int center_y, int width, int height) {
    // 获取图片高度和宽度
    int image_height = image.rows;
    int image_width = image.cols;

    // 计算裁剪区域
    int x1 = std::max(0, static_cast<int>(center_x - width / 2.0));
    int y1 = std::max(0, static_cast<int>(center_y - height / 2.0));
    int x2 = std::min(image_width, x1 + width);
    int y2 = std::min(image_height, y1 + height);

    // 创建全黑背景
    cv::Mat black_image(height, width, CV_8UC1, cv::Scalar(0));

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

static unsigned char *load_model(const char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);

    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *) malloc(model_len);

    fseek(fp, 0, SEEK_SET);

    if (model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }

    *model_size = model_len;

    if (fp) {
        fclose(fp);
    }
    return model;
}

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf(
            "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, "
            "size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
            attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
            get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


RKNNModel_::RKNNModel_(std::string path) {
    model = load_model(path.c_str(), &model_len);
    int ret = rknn_init(&ctx, model, model_len, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
    }

    // Get Model Input Output Info
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
    }

    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
        }
        dump_tensor_attr(&(output_attrs[i]));
    }
}

int RKNNModel_::set_input(rknn_input *inputs) {
    int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    return 0;
}

void RKNNModel_::get_output(rknn_output *outputs) {
    int ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
    }
}

int RKNNModel_::run(rknn_core_mask mask) {
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    int ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0) {
        printf("rknn_CORE_MASK fail! ret=%d\n", ret);
        return -1;
    }

    auto start = std::chrono::system_clock::now();

    // Run
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "network:" << double(duration.count()) * std::chrono::microseconds::period::num /
                               std::chrono::microseconds::period::den * 1000 << std::endl;

    return 0;
}

int RKNNModel_::get_num_output() {
    return io_num.n_output;
}

void RKNNModel_::release_output(rknn_output *outputs) {
    rknn_outputs_release(ctx, io_num.n_output, outputs);
}


SuperPoint::SuperPoint(int _width, int _height, std::string path) {
    width = _width;
    height = _height;

    assert(width % 8 == 0);
    assert(height % 8 == 0);

    model = std::make_shared<RKNNModel_>(path);
    maxpool = new float[height * width];
    softmax_sum = new float[height * width];
    l2_sum = new float[height * width];
}

std::vector<cv::KeyPoint> SuperPoint::nms(float *pool_t, int window_size, float threshold, int borders) {
    int window_center = window_size / 2;
    int number = 0;
    std::vector<cv::KeyPoint> point_array;
    for (int i = borders; i < height - borders; i++) {
        for (int j = borders; j < width - borders; j++) {
            if (threshold > maxpool[i * width + j])
                continue;

            int largest = true;
            for (int x = -window_center; x < window_center; x++) {
                if (!largest)
                    break;
                for (int y = -window_center; y < window_center; y++) {
                    if (!largest)
                        break;
                    if (x == 0 and y == 0)
                        continue;
                    if (i + x >= 0 && i + x < height && j + y >= 0 && j + y < width) {
                        if (maxpool[(i + x) * width + j + y] >= maxpool[i * width + j]) {
                            largest = false;
                        }
                    }
                }
            }
            if (largest) {
                point_array.push_back(cv::KeyPoint(cv::Point2f(j, i), 8, -1, maxpool[i * width + j]));
            }
        }
    }
    return point_array;
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> SuperPoint::detect(cv::Mat img) {
    memset(maxpool, 0, height * width * sizeof(float));
    memset(softmax_sum, 0, height / 8 * width / 8 * sizeof(float));
    memset(l2_sum, 0, height / 8 * width / 8 * sizeof(float));

    assert(img.rows == height);
    assert(img.cols == width);
    assert(img.channels() == 1);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols * img.rows * img.channels() * sizeof(uint8_t);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    model->set_input(inputs);
    model->run();

    rknn_output outputs[model->get_num_output()];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < model->get_num_output(); i++) {
        outputs[i].want_float = 1;
    }

    model->get_output(outputs);

    float *score = (float *) outputs[0].buf;
    float *des = (float *) outputs[1].buf;

    for (int k = 0; k < 65; k++) {
        for (int j = 0; j < height / 8; j++) {
            for (int i = 0; i < width / 8; i++) {
                float now = std::exp(score[k * height / 8 * width / 8 + j * width / 8 + i]);
                int k_w = k % 8, k_h = k / 8;
                softmax_sum[j * width / 8 + i] = softmax_sum[j * width / 8 + i] + now;
                if (k != 64) {
                    maxpool[(j * 8 + k_h) * width + (i * 8 + k_w)] = now;
                }
            }
        }
    }

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            maxpool[j * width + i] = maxpool[j * width + i] / softmax_sum[(j / 8) * width / 8 + (i / 8)];
        }
    }

    for (int k = 0; k < 256; k++) {
        for (int j = 0; j < height / 8; j++) {
            for (int i = 0; i < width / 8; i++) {
                l2_sum[j * width / 8 + i] += des[k * height / 8 * width / 8 + j * width / 8 + i] *
                                             des[k * height / 8 * width / 8 + j * width / 8 + i];
            }
        }
    }

    std::vector<cv::KeyPoint> keypoint = nms(maxpool);
    std::sort(keypoint.begin(), keypoint.end(), [](cv::KeyPoint x, cv::KeyPoint y) { return x.response > y.response; });
    cv::Mat feature = cv::Mat::zeros(keypoint.size(), 256, CV_32F);
    for (int i = 0; i < keypoint.size(); i++) {
        auto &point = keypoint[i].pt;
        int x1 = ((int) point.x) / 8, x2 = ((int) point.x) / 8 + 1;
        int y1 = ((int) point.y) / 8, y2 = ((int) point.y) / 8 + 1;
        float x = point.x / 8, y = point.y / 8;

        float sum = 0;
        for (int k = 0; k < 256; k++) {

            feature.at<float>(i, k) =
                    (des[k * width / 8 * height / 8 + y1 * width / 8 + x1] / std::sqrt(l2_sum[y1 * width / 8 + x1])) *
                    (x2 - x) * (y2 - y) +
                    (des[k * width / 8 * height / 8 + y1 * width / 8 + x2] / std::sqrt(l2_sum[y1 * width / 8 + x2])) *
                    (x - x1) * (y2 - y) +
                    (des[k * width / 8 * height / 8 + y2 * width / 8 + x1] / std::sqrt(l2_sum[y2 * width / 8 + x1])) *
                    (x2 - x) * (y - y1) +
                    (des[k * width / 8 * height / 8 + y2 * width / 8 + x2] / std::sqrt(l2_sum[y2 * width / 8 + x2])) *
                    (x - x1) * (y - y1);
            sum += feature.at<float>(i, k) * feature.at<float>(i, k);
        }
        for (int k = 0; k < 256; k++) {
            feature.at<float>(i, k) /= std::sqrt(sum);
        }
    }

    model->release_output(outputs);

    return std::make_tuple(keypoint, feature);
}


float logSigmoid(float x) {
    return std::log(1 / (1 + std::exp(-x)));
}

LightglueMatch::LightglueMatch(std::string model_superpoint_path, std::string model_lightglue_path, int point) {
    model_superpoint = std::make_shared<SuperPoint>(960, 544, "../resource/superpoint.rknn");
    model_lightglue = std::make_shared<RKNNModel_>("../resource/lightglue_3layers.rknn");
    work_loop_thread_ptr = std::make_shared<std::thread>(&LightglueMatch::work_loop, this);
    system_state.store(0);
    HIT_state.store(0);
    _async_flag.store(0);
    point_num = 256;
    RUN_number = 0;
    loss_number = 0;

    inference_w = 960;
    inference_h = 544;

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

}


void LightglueMatch::work_loop() {
    while (true) {

        if (system_state.load() != 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {

            _async_flag.store(1);   //标记系统正在运行

            cv::Mat GRAY_frame, GRAY_satellite;
            cv::cvtColor(_update_frame, GRAY_frame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(_satellite, GRAY_satellite, cv::COLOR_BGR2GRAY);

            if (ONLY_SP) {
//                resize 相关工作
                float resize_rate_w_satellite = _satellite.cols / 960;
                float resize_rate_h_satellite = _satellite.rows / 544;

                float resize_rate_w_update_frame = _update_frame.cols / 960;
                float resize_rate_h_update_frame = _update_frame.rows / 544;

//               需要把输入的敌点也resize 关于 卫星图的输入 所以需要使用卫星图的尺寸；
                aim_point.x = aim_point.x / resize_rate_w_satellite;
                aim_point.y = aim_point.y / resize_rate_h_satellite;

                cv::Mat resize_frame_GRAY, resize_satellite_GRAY;
                cv::resize(GRAY_frame, resize_frame_GRAY,
                           cv::Size(inference_w, inference_h));
                cv::resize(GRAY_satellite, resize_satellite_GRAY,
                           cv::Size(inference_w, inference_h));
                std::pair<cv::Mat, bool> pai_SP = superPoint_inference(resize_frame_GRAY, resize_satellite_GRAY);
                if (pai_SP.second) //标记运行结果
                {
                    //预测点
                    std::vector<cv::Point2f> _srcPoints(1, aim_point);
                    std::vector<cv::Point2f> _dstPoints;
                    cv::perspectiveTransform(_srcPoints, _dstPoints, pai_SP.first);//转换到 弹 的视角
//
                    aim_point_detect.x = _dstPoints[0].x * resize_rate_w_update_frame;//弹敌点缩放
                    aim_point_detect.y = _dstPoints[0].y * resize_rate_h_update_frame;
                    _async_flag.store(2);
                    HIT_state.store(1);
                } else {
                    _async_flag.store(0);
                    loss_number += 1;
                }

                system_state.store(0);
            } else {
                float crop_cent_x, h_rate, crop_cent_y;
                if (HIT_state.load() == 1) {
                    crop_cent_x = aim_point_detect.x;
                    crop_cent_y = aim_point_detect.y;
                    std::cout << "匹配命中切图 crop_cent_y:" << crop_cent_y << std::endl;
                } else {
                    crop_cent_x = GRAY_frame.cols / 2;

                    h_rate = GRAY_frame.rows / 2;//此处为滑动起点 可变

                    crop_cent_y = (RUN_number % 2) * h_rate + 272;
                    std::cout << "搜索模式切图 crop_cent_y:" << crop_cent_y << std::endl;
                }

                std::pair<cv::Mat, cv::Point2f> crop_update_frame = crop_and_pad_image(GRAY_frame,
                                                                                       int(crop_cent_x),
                                                                                       crop_cent_y,
                                                                                       inference_w, inference_h);

                std::pair<cv::Mat, cv::Point2f> crop_satellite = crop_and_pad_image(GRAY_satellite, aim_point.x,
                                                                                    aim_point.y,
                                                                                    inference_w, inference_h);

                std::pair<cv::Mat, bool> pai_LG = lightGlue_inference(crop_satellite.first, crop_update_frame.first,
                                                                      crop_satellite.second, crop_update_frame.second);


                if (pai_LG.second) //标记运行结果
                {
                    //预测点
                    std::vector<cv::Point2f> _srcPoints(1, aim_point);
                    std::vector<cv::Point2f> _dstPoints;
                    cv::perspectiveTransform(_srcPoints, _dstPoints, pai_LG.first);

                    aim_point_detect.x = _dstPoints[0].x;
                    aim_point_detect.y = _dstPoints[0].y;
                    _async_flag.store(2);
                    HIT_state.store(1);
                    loss_number = 0;

                } else {
                    _async_flag.store(0);
                    loss_number += 1;
                }
                system_state.store(0);
            }
            system_state.store(0);
        }
    }
}

void LightglueMatch::async(cv::Mat satellite, cv::Mat frame, cv::Point _aim_point, bool _ONLY_SP) {
    _update_frame = frame;
    _satellite = satellite;
    aim_point = _aim_point;
    ONLY_SP.store(_ONLY_SP);
    RUN_number += 1;
    _async_flag.store(1);   //标记系统正在运行
    system_state.store(1);

    if (loss_number % 5 == 0) {
        HIT_state.store(0); //丢失五次进入搜索模式；搜索模式 corp区域将上下滑动；
    }

}

int LightglueMatch::get_statu() {
    return _async_flag.load();
}

cv::Point LightglueMatch::syncronize() {
    return aim_point_detect;
}

std::pair<cv::Mat, bool>
LightglueMatch::lightGlue_inference(cv::Mat img0, cv::Mat img1, cv::Point2f img0_cut_point,
                                    cv::Point2f img1_cut_point) {

    float max_indice_x[point_num], max_indice_y[point_num];
    int indice_x[point_num], indice_y[point_num];

    std::vector<cv::KeyPoint> keypoint0, keypoint1;
    cv::Mat feature0, feature1;
    std::tie(keypoint0, feature0) = model_superpoint->detect(img0);
    std::tie(keypoint1, feature1) = model_superpoint->detect(img1);

    float kpts0[point_num * 2], kpts1[point_num * 2], scores[point_num * point_num];

    if (keypoint0.size() < point_num || keypoint1.size() < point_num) {
        cv::Mat H;
        return std::make_pair(H, false);
    }

    if (img0_cut_point.x != 0) {
        for (cv::KeyPoint &kp: keypoint0) {
            kp.pt.x += img0_cut_point.x;
            kp.pt.y += img0_cut_point.y;
        }
    }

    if (img1_cut_point.x != 0) {
        for (cv::KeyPoint &kp: keypoint1) {
            kp.pt.x += img1_cut_point.x;
            kp.pt.y += img1_cut_point.y;
        }
    }


    for (int i = 0; i < point_num; i++) {
        kpts0[i * 2 + 0] = (keypoint0[i].pt.x - 960 / 2) / 480;
        kpts0[i * 2 + 1] = (keypoint0[i].pt.y - 544 / 2) / 480;
    }

    for (int i = 0; i < point_num; i++) {
        kpts1[i * 2 + 0] = (keypoint1[i].pt.x - 960 / 2) / 480;
        kpts1[i * 2 + 1] = (keypoint1[i].pt.y - 544 / 2) / 480;
    }

    rknn_input inputs[4];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = point_num * 2 * sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[0].buf = kpts0;

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].size = point_num * 2 * sizeof(float);
    inputs[1].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[1].buf = kpts1;

    inputs[2].index = 2;
    inputs[2].type = RKNN_TENSOR_FLOAT32;
    inputs[2].size = point_num * 256 * sizeof(float);
    inputs[2].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[2].buf = feature0.data;

    inputs[3].index = 3;
    inputs[3].type = RKNN_TENSOR_FLOAT32;
    inputs[3].size = point_num * 256 * sizeof(float);
    inputs[3].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[3].buf = feature1.data;

    model_lightglue->set_input(inputs);

    model_lightglue->run();

    rknn_output outputs[model_lightglue->get_num_output()];


    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < model_lightglue->get_num_output(); i++) {
        outputs[i].want_float = 1;
    }

    model_lightglue->get_output(outputs);

    float *desc0 = (float *) outputs[0].buf;
    float *z0 = (float *) outputs[1].buf;
    float *z1 = (float *) outputs[2].buf;

    for (int i = 0; i < point_num; i++) {
        for (int j = 0; j < point_num; j++) {
            scores[i * point_num + j] = desc0[i * point_num + j] + logSigmoid(z0[i]) + logSigmoid(z1[j]);
        }
    }

    for (int i = 0; i < point_num; i++) {
        for (int j = 0; j < point_num; j++) {
            if (j == 0) {
                max_indice_x[i] = scores[i * point_num + j];
                indice_x[i] = 0;
            } else if (max_indice_x[i] < scores[i * point_num + j]) {
                max_indice_x[i] = scores[i * point_num + j];
                indice_x[i] = j;
            }
        }
    }

    for (int j = 0; j < point_num; j++) {
        for (int i = 0; i < point_num; i++) {
            if (i == 0) {
                max_indice_y[j] = scores[i * point_num + j];
                indice_y[j] = 0;
            } else if (max_indice_y[j] < scores[i * point_num + j]) {
                max_indice_y[j] = scores[i * point_num + j];
                indice_y[j] = i;
            }
        }
    }

    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts0, pts1;

    for (int i = 0; i < point_num; i++) {
        if (indice_y[indice_x[i]] == i && std::exp(scores[i * point_num + indice_x[i]]) > 0.1
            && std::exp(scores[i * point_num + indice_x[i]]) > 0.7 * std::exp(scores[indice_x[i] * point_num + i])) {
            good_matches.push_back(cv::DMatch(i, indice_x[i], std::exp(scores[i * point_num + indice_x[i]])));
            pts0.push_back(keypoint0[i].pt);
            pts1.push_back(keypoint1[indice_x[i]].pt);
        }
    }

    if (pts0.size() < 4 || pts1.size() < 4) {
        cv::Mat H;
        return std::make_pair(H, false);
    }

    cv::Mat H, mask;
    H = cv::findHomography(pts0, pts1, cv::RANSAC, 3.0, mask);

//#ifdef MATCH_DEBUG
//    cv::Mat img_matches;
//    cv::drawMatches(img0, keypoint0, img1, keypoint1, good_matches, img_matches, cv::Scalar::all(-1),
//                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    cv::imshow("LG_matches", img_matches);
//    cv::waitKey(100);
//
//#endif
    return std::make_pair(H, true);
}

std::pair<cv::Mat, bool> LightglueMatch::superPoint_inference(cv::Mat img0, cv::Mat img1) {

    std::vector<cv::KeyPoint> keypoint0, keypoint1;
    cv::Mat feature0, feature1;


    std::tie(keypoint0, feature0) = model_superpoint->detect(img0);
    std::tie(keypoint1, feature1) = model_superpoint->detect(img1);
    std::vector<std::vector<cv::DMatch>> knn_matches;

    int rows_to_copy = std::min(feature0.rows, 500); // 确保不会超出矩阵实际大小
    cv::Mat subset_0(feature0, cv::Range(0, rows_to_copy), cv::Range::all()); // 截取前500行

    rows_to_copy = std::min(feature1.rows, 500); // 确保不会超出矩阵实际大小
    cv::Mat subset_1(feature1, cv::Range(0, rows_to_copy), cv::Range::all()); // 截取前500行

    matcher->knnMatch(subset_0, subset_1, knn_matches, 2);

    const float ratio_thresh = 0.8f;
    std::vector<cv::Point2f> pts0, pts1;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            pts0.push_back(keypoint0[knn_matches[i][0].queryIdx].pt);
            pts1.push_back(keypoint1[knn_matches[i][0].trainIdx].pt);
        }
    }

    if (pts0.size() < 30) {
        cv::Mat H;
        knn_matches.clear();
        return std::make_pair(H, false);
    }

    cv::Mat H, mask;
    H = cv::findHomography(pts0, pts1, cv::RANSAC, 3.0, mask);

//#ifdef MATCH_DEBUG
//    cv::Mat img_matches;
//    cv::drawMatches(img0, keypoint0, img1, keypoint1, knn_matches, img_matches, cv::Scalar::all(-1),
//                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    cv::imshow("SP_matches", img_matches);
//    cv::waitKey(100);
//
//#endif


    int matchedPointsCount = cv::countNonZero(mask);

    if (matchedPointsCount > 30) {
        knn_matches.clear();
        return std::make_pair(H, true);
    } else {
        cv::Mat H;

        knn_matches.clear();

        return std::make_pair(H, false);
    }

}
