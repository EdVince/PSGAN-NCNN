#include "psgan.h"
#include "cpu.h"

PSGAN::PSGAN()
{
    num_thread = ncnn::get_big_cpu_count();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(num_thread);

    FaceParserNet.load_param("assert/faceparser-sim-opt.param");
    FaceParserNet.load_model("assert/faceparser-sim-opt.bin");

    LandmarkNet.load_param("assert/slim_160_latest-opt.param");
    LandmarkNet.load_model("assert/slim_160_latest-opt.bin");

    forward1Net.load_param("assert/forward1-sim-opt.param");
    forward1Net.load_model("assert/forward1-sim-opt.bin");

    forward2Net.load_param("assert/forward2-sim-opt.param");
    forward2Net.load_model("assert/forward2-sim-opt.bin");
}
PSGAN::~PSGAN()
{
    FaceParserNet.clear();
    LandmarkNet.clear();
    forward1Net.clear();
    forward2Net.clear();
}


std::vector<pair<cv::Point, cv::Point>> PSGAN::detect(cv::Mat image)
{
    std::vector<pair<cv::Point, cv::Point>> faces;
    cv::Mat frame = image.clone();
    std::vector<FaceInfo> face_info;

    ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB2BGR, frame.cols, frame.rows);
    lffd.detect(inmat, face_info, 240, 320);

    for (int i = 0; i < face_info.size(); i++)
    {
        auto face = face_info[i];
        faces.push_back(make_pair(cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2)));
    }

    return faces;
}

pair<cv::Point, cv::Point> PSGAN::crop(cv::Mat& image, pair<cv::Point, cv::Point>& face, double up_ratio, double down_ratio, double width_ratio)
{
    int width = image.cols, height = image.rows;
    int face_height = face.second.y - face.first.y, face_width = face.second.x - face.first.x;
    double delta_up = up_ratio * face_height, delta_down = down_ratio * face_height, delta_width = width_ratio * width;
    int img_left = int(max(0.0, face.first.x - delta_width)), img_top = int(max(0.0, face.first.y - delta_up));
    int img_right = int(min((double)width, face.second.x + delta_width)), img_bottom = int(min((double)height, face.second.y + delta_down));
    image = image(cv::Rect(cv::Point(img_left, img_top), cv::Point(img_right, img_bottom))).clone();
    face = make_pair(cv::Point(face.first.x - img_left, face.first.y - img_top), cv::Point(face.second.x - img_left, face.second.y - img_top));
    pair<cv::Point, cv::Point> face_expand = make_pair(cv::Point(img_left, img_top), cv::Point(img_right, img_bottom));
    cv::Point center(int(face_expand.first.x + face_expand.second.x + 0.5) / 2, int(face_expand.first.y + face_expand.second.y + 0.5) / 2);
    width = image.cols; height = image.rows;
    int crop_left = img_left, crop_top = img_top, crop_right = img_right, crop_bottom = img_bottom;
    if (width > height) {
        int left = int(center.x - height / 2), right = int(center.x + height / 2);
        if (left < 0) {
            left = 0;
            right = height;
        }
        else if (right > width) {
            left = width - height;
            right = width;
        }
        image = image(cv::Rect(cv::Point(left, 0), cv::Point(right, height))).clone();
        face = make_pair(cv::Point(face.first.x - left, face.first.y), cv::Point(face.second.x - left, face.second.y));
        crop_left += left;
        crop_right = crop_left + height;
    }
    else {
        int top = int(center.y - width / 2), bottom = int(center.y + width / 2);
        if (top < 0) {
            top = 0;
            bottom = width;
        }
        else if (bottom > height) {
            top = height - width;
            bottom = height;
        }
        image = image(cv::Rect(cv::Point(0, top), cv::Point(width, bottom))).clone();
        face = make_pair(cv::Point(face.first.x, face.first.y - top), cv::Point(face.second.x, face.second.y - top));
        crop_top += top;
        crop_bottom = crop_top + width;
    }
    pair<cv::Point, cv::Point> crop_face = make_pair(cv::Point(crop_left, crop_top), cv::Point(crop_right, crop_bottom));
    return crop_face;
}

cv::Mat PSGAN::faceparser(const cv::Mat image)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, 512, 512);
    in.substract_mean_normalize(FaceParserNet_mean_vals, FaceParserNet_norm_vals);

    ncnn::Extractor ex = FaceParserNet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(num_thread);
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);

    int mapper[] = { 0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0 };
    std::vector<float*> p_tmp(19);
    std::vector<float> v_tmp(19);
    cv::Mat mask(cv::Size(512, 512), CV_8UC1);
    for (int i = 0; i < 19; i++)
        p_tmp[i] = out.channel(i);
    for (int i = 0; i < 512; i++) {
        for (int j = 0; j < 512; j++) {
            for (int k = 0; k < 19; k++) {
                v_tmp[k] = p_tmp[k][0];
                p_tmp[k]++;
            }
            mask.at<uchar>(i, j) = mapper[(int)std::distance(v_tmp.begin(), std::max_element(v_tmp.begin(), v_tmp.end()))];
        }
    }

    return mask;
}

std::vector<std::vector<int>> PSGAN::landmark(const cv::Mat image, pair<cv::Point, cv::Point> face)
{
    cv::Mat srcROI = image(cv::Rect(cv::Point(face.first.x, face.first.y), cv::Point(face.second.x, face.second.y))).clone();
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(srcROI.data, ncnn::Mat::PIXEL_RGB2BGR, srcROI.cols, srcROI.rows, 160, 160);
    in.substract_mean_normalize(LandmarkNet_mean_vals, LandmarkNet_norm_vals);

    ncnn::Extractor ex = LandmarkNet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(num_thread);
    ex.input("input1", in);
    ncnn::Mat keyPoint;
    ex.extract("output1", keyPoint);

    std::vector<cv::Point2f> faceLandmark(68);
    for (int i = 0; i < 68; i++)
        faceLandmark[i] = cv::Point2f(keyPoint[2 * i] * srcROI.cols + face.first.x, keyPoint[2 * i + 1] * srcROI.rows + face.first.y);

    std::vector<std::vector<int>> lms(68);
    for (int i = 0; i < 68; i++) {
        lms[i] = { (int)std::round(faceLandmark[i].y * 256.0 / image.cols), (int)std::round(faceLandmark[i].x * 256.0 / image.cols) };
    }

    return lms;
}

void PSGAN::copy_area(cv::Mat& tar, cv::Mat& src, std::vector<std::vector<int>> lms)
{
    std::vector<int> lms_0(lms.size()), lms_1(lms.size());
    for (int i = 0; i < lms.size(); i++) {
        lms_0[i] = lms[i][0];
        lms_1[i] = lms[i][1];
    }
    auto lms_0_minmax = std::minmax_element(lms_0.begin(), lms_0.end());
    auto lms_1_minmax = std::minmax_element(lms_1.begin(), lms_1.end());

    int rect[] = { *lms_1_minmax.first - 16,*lms_0_minmax.first - 16,*lms_1_minmax.second + 16 + 1,*lms_0_minmax.second + 16 + 1 };

    for (int i = rect[1]; i < rect[3]; i++) {
        for (int j = rect[0]; j < rect[2]; j++) {
            tar.at<double>(i, j) = src.at<double>(i, j);
            src.at<double>(i, j) = 0;
        }
    }
}

void PSGAN::process(cv::Mat mask, std::vector<std::vector<int>> lms, std::vector<cv::Mat>& mask_aug, std::vector<std::vector<cv::Mat>>& diff_re)
{
    cv::Mat ys(cv::Size(256, 256), CV_64FC1);
    for (int i = 0; i < ys.rows; i++)
        for (int j = 0; j < ys.cols; j++)
            ys.at<double>(i, j) = i;
    cv::Mat xs(cv::Size(256, 256), CV_64FC1);
    for (int i = 0; i < xs.rows; i++)
        for (int j = 0; j < xs.cols; j++)
            xs.at<double>(i, j) = j;
    std::vector<cv::Mat> diff(68 * 2);
    for (int i = 0; i < 68; i++)
        diff[i] = ys - lms[i][0];
    for (int i = 0; i < 68; i++)
        diff[68 + i] = xs - lms[i][1];

    std::vector<std::vector<int>> lms_eye_left(lms.begin() + 42, lms.begin() + 48);
    std::vector<std::vector<int>> lms_eye_right(lms.begin() + 36, lms.begin() + 42);

    cv::Mat mask_lip(mask.size(), CV_64FC1);
    for (int i = 0; i < mask_lip.rows; i++) {
        for (int j = 0; j < mask_lip.cols; j++) {
            uchar value = mask.at<uchar>(i, j);
            mask_lip.at<double>(i, j) = ((value == 7) ? 1 : 0) + ((value == 9) ? 1 : 0);
        }
    }

    cv::Mat mask_face(mask.size(), CV_64FC1);
    for (int i = 0; i < mask_face.rows; i++) {
        for (int j = 0; j < mask_face.cols; j++) {
            uchar value = mask.at<uchar>(i, j);
            mask_face.at<double>(i, j) = ((value == 1) ? 1 : 0) + ((value == 6) ? 1 : 0);
        }
    }

    cv::Mat mask_eyes = cv::Mat::zeros(mask.size(), CV_64FC1);
    copy_area(mask_eyes, mask_face, lms_eye_left);
    copy_area(mask_eyes, mask_face, lms_eye_right);

    mask_aug = std::vector<cv::Mat>{ mask_lip.clone(), mask_face.clone(), mask_eyes.clone() };
    cv::Mat mask_lip_small, mask_face_small, mask_eyes_small;
    cv::resize(mask_lip, mask_lip_small, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
    cv::resize(mask_face, mask_face_small, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
    cv::resize(mask_eyes, mask_eyes_small, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
    std::vector<std::vector<cv::Mat>> mask_re(std::vector<std::vector<cv::Mat>>(3, std::vector<cv::Mat>(136)));
    for (int i = 0; i < 136; i++) {
        mask_re[0][i] = mask_lip_small.clone();
        mask_re[1][i] = mask_face_small.clone();
        mask_re[2][i] = mask_eyes_small.clone();
    }
    diff_re = std::vector<std::vector<cv::Mat>>(3, std::vector<cv::Mat>(136));
    for (int i = 0; i < 136; i++) {
        cv::resize(diff[i], diff[i], cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
        diff_re[0][i] = diff[i].clone();
        diff_re[1][i] = diff[i].clone();
        diff_re[2][i] = diff[i].clone();
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 136; j++) {
            diff_re[i][j] = diff_re[i][j].mul(mask_re[i][j]);
        }
    }
    std::vector<double> tmp(136);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 64; k++) {
                for (int n = 0; n < 136; n++) {
                    tmp[n] = std::abs(diff_re[i][n].at<double>(j, k));
                    tmp[n] = tmp[n] * tmp[n];
                }
                double norm = std::sqrt(std::accumulate(tmp.begin(), tmp.end(), 0));
                if (norm == 0) {
                    norm = 1e10;
                }
                for (int n = 0; n < 136; n++) {
                    diff_re[i][n].at<double>(j, k) /= norm;
                }
            }
        }
    }
}

void PSGAN::preprocess(const cv::Mat in_image, cv::Mat& image, std::vector<cv::Mat>& mask_aug, std::vector<std::vector<cv::Mat>>& diff_re, pair<cv::Point, cv::Point>& crop_face)
{
    image = in_image.clone();
    std::vector<pair<cv::Point, cv::Point>> faces = detect(image);
    pair<cv::Point, cv::Point> face_on_image = faces[0], face = faces[0];
    crop_face = crop(image, face, 0.6 / 0.85, 0.2 / 0.85, 0.2 / 0.85);
    cv::Mat mask = faceparser(image);
    cv::resize(mask, mask, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
    std::vector<std::vector<int>> lms = landmark(image, face);
    process(mask, lms, mask_aug, diff_re);
    cv::resize(image, image, cv::Size(256, 256));
}

ncnn::Mat PSGAN::from_64_64_136(std::vector<cv::Mat> in)
{
    ncnn::Mat out(64, 64, 136);
    for (int i = 0; i < 136; i++) {
        double* data = (double*)in[i].data;
        float* pt = out.channel(i);
        for (int j = 0; j < 64 * 64; j++) {
            *pt = *data;
            pt++;
            data++;
        }
    }
    return out;
}

ncnn::Mat PSGAN::from_256_256_1(cv::Mat in)
{
    ncnn::Mat out(256, 256, 1);
    double* data = (double*)in.data;
    float* pt = out.channel(0);
    for (int j = 0; j < 256 * 256; j++) {
        *pt = *data;
        pt++;
        data++;
    }
    return out;
}

void PSGAN::postprocess(cv::Mat source, cv::Mat& result)
{
    int height = source.rows, width = source.cols;
    cv::Mat small_source;
    cv::resize(source, small_source, cv::Size(256, 256));

    cv::Mat sourceF, small_sourceF;
    source.convertTo(sourceF, CV_32F);

    cv::Mat big_small_source;
    cv::resize(small_source, big_small_source, cv::Size(width, height));
    big_small_source.convertTo(small_sourceF, CV_32F);

    cv::Mat laplacian_diff = sourceF - small_sourceF;

    cv::Mat big_result;
    cv::resize(result, big_result, cv::Size(width, height));
    cv::Mat resultF;
    big_result.convertTo(resultF, CV_32F);
    resultF += laplacian_diff;

    resultF.convertTo(result, CV_8U);
}

cv::Mat PSGAN::go(const cv::Mat _source, const cv::Mat _reference)
{
    cv::Mat source = _source.clone();
    cv::Mat reference = _reference.clone();

    cv::Mat real_A, real_B;
    std::vector<cv::Mat> mask_A, mask_B;
    std::vector<std::vector<cv::Mat>> diff_A, diff_B;
    pair<cv::Point, cv::Point> crop_face_A, crop_face_B;
    preprocess(source, real_A, mask_A, diff_A, crop_face_A);
    preprocess(reference, real_B, mask_B, diff_B, crop_face_B);

    ncnn::Mat ncnn_real_A = ncnn::Mat::from_pixels(real_A.data, ncnn::Mat::PIXEL_RGB, real_A.cols, real_A.rows);
    ncnn::Mat ncnn_real_B = ncnn::Mat::from_pixels(real_B.data, ncnn::Mat::PIXEL_RGB, real_B.cols, real_B.rows);
    ncnn_real_A.substract_mean_normalize(forward1Net_mean_vals, forward1Net_norm_vals);
    ncnn_real_B.substract_mean_normalize(forward1Net_mean_vals, forward1Net_norm_vals);
    ncnn::Mat ncnn_mask_A_0 = from_256_256_1(mask_A[0]);
    ncnn::Mat ncnn_mask_A_1 = from_256_256_1(mask_A[1]);
    ncnn::Mat ncnn_mask_A_2 = from_256_256_1(mask_A[2]);
    ncnn::Mat ncnn_mask_B_0 = from_256_256_1(mask_B[0]);
    ncnn::Mat ncnn_mask_B_1 = from_256_256_1(mask_B[1]);
    ncnn::Mat ncnn_mask_B_2 = from_256_256_1(mask_B[2]);
    ncnn::Mat ncnn_diff_A_0 = from_64_64_136(diff_A[0]);
    ncnn::Mat ncnn_diff_A_1 = from_64_64_136(diff_A[1]);
    ncnn::Mat ncnn_diff_A_2 = from_64_64_136(diff_A[2]);
    ncnn::Mat ncnn_diff_B_0 = from_64_64_136(diff_B[0]);
    ncnn::Mat ncnn_diff_B_1 = from_64_64_136(diff_B[1]);
    ncnn::Mat ncnn_diff_B_2 = from_64_64_136(diff_B[2]);

    ncnn::Extractor ex1 = forward1Net.create_extractor();
    ex1.set_light_mode(true);
    ex1.set_num_threads(num_thread);
    ex1.input("real_A", ncnn_real_A);
    ex1.input("real_B", ncnn_real_B);
    ex1.input("mask_A_0", ncnn_mask_A_0);
    ex1.input("mask_A_1", ncnn_mask_A_1);
    ex1.input("mask_A_2", ncnn_mask_A_2);
    ex1.input("mask_B_0", ncnn_mask_B_0);
    ex1.input("mask_B_1", ncnn_mask_B_1);
    ex1.input("mask_B_2", ncnn_mask_B_2);
    ex1.input("diff_A_0", ncnn_diff_A_0);
    ex1.input("diff_A_1", ncnn_diff_A_1);
    ex1.input("diff_A_2", ncnn_diff_A_2);
    ex1.input("diff_B_0", ncnn_diff_B_0);
    ex1.input("diff_B_1", ncnn_diff_B_1);
    ex1.input("diff_B_2", ncnn_diff_B_2);

    ncnn::Mat gamma, beta;
    ex1.extract("gamma", gamma);
    ex1.extract("beta", beta);

    ncnn::Extractor ex2 = forward2Net.create_extractor();
    ex2.set_light_mode(true);
    ex2.set_num_threads(num_thread);
    ex2.input("real_A", ncnn_real_A);
    ex2.input("gamma", gamma);
    ex2.input("beta", beta);

    ncnn::Mat ncnn_fake_A;
    ex2.extract("fake_A", ncnn_fake_A);

    auto minmax0 = std::minmax_element((float*)ncnn_fake_A.channel(0), (float*)ncnn_fake_A.channel(0) + 256 * 256);
    auto minmax1 = std::minmax_element((float*)ncnn_fake_A.channel(1), (float*)ncnn_fake_A.channel(1) + 256 * 256);
    auto minmax2 = std::minmax_element((float*)ncnn_fake_A.channel(2), (float*)ncnn_fake_A.channel(2) + 256 * 256);
    float min_ = std::min((*minmax0.first), std::min((*minmax1.first), (*minmax2.first)));
    float max_ = std::max((*minmax0.second), std::max((*minmax1.second), (*minmax2.second)));
    float ratio = 1.f / (max_ - min_);
    float mean2[3] = { min_,min_,min_ };
    float norm2[3] = { ratio, ratio, ratio };
    ncnn_fake_A.substract_mean_normalize(mean2, norm2);
    float norm3[3] = { 255.f,255.f,255.f };
    ncnn_fake_A.substract_mean_normalize(0, norm3);

    cv::Mat fake_A(cv::Size(256, 256), CV_8UC3);
    ncnn_fake_A.to_pixels(fake_A.data, ncnn::Mat::PIXEL_RGB);

    cv::Mat source_crop = source(cv::Rect(cv::Point(crop_face_A.first.x, crop_face_A.first.y), cv::Point(crop_face_A.second.x, crop_face_A.second.y))).clone();

    postprocess(source_crop, fake_A);

    return fake_A;
}