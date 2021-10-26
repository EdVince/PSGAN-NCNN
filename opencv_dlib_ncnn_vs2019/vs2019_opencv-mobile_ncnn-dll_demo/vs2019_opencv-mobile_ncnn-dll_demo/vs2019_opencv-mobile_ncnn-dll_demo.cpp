#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>
#include <stdio.h>

#include "net.h"

#include <opencv2/opencv.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

using namespace std;

bool debug = true;

dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
dlib::shape_predictor predictor;

cv::Mat resize_by_max(cv::Mat image, int max_side=512, bool force=false)
{
    cv::Mat img;
    int h = image.rows, w = image.cols;

    if ((max(h, w) < max_side) && (!force)) {
        img = image.clone();
    }
    else {
        double ratio = (double)max(h, w) / max_side;
        w = int(w / ratio + 0.5);
        h = int(h / ratio + 0.5);
        cv::resize(image, img, cv::Size(w, h));
    }
    return img;
}

std::vector<dlib::rectangle> detect(cv::Mat image)
{
    int h = image.rows, w = image.cols;
    image = resize_by_max(image, 361);
    cv::resize(image, image, cv::Size(2 * image.cols, 2 * image.rows));
    int actual_h = image.rows, actual_w = image.cols;
    dlib::matrix<dlib::rgb_pixel> dlib_image;
    dlib::assign_image(dlib_image, dlib::cv_image<dlib::rgb_pixel>(image));
    std::vector<dlib::rectangle> faces_on_small = detector(dlib_image);
    std::vector<dlib::rectangle> faces;
    for (auto face : faces_on_small) {
        faces.push_back(dlib::rectangle(
            int((double)face.left() / actual_w * w + 0.5),
            int((double)face.top() / actual_h * h + 0.5),
            int((double)face.right() / actual_w * w + 0.5),
            int((double)face.bottom() / actual_h * h + 0.5)));
    }
    return faces;
}

dlib::rectangle crop(cv::Mat& image, dlib::rectangle& face, double up_ratio, double down_ratio, double width_ratio)
{
    int width = image.cols, height = image.rows;
    int face_height = face.height(), face_width = face.width();
    double delta_up = up_ratio * face_height, delta_down = down_ratio * face_height, delta_width = width_ratio * width;
    int img_left = int(max(0.0, face.left() - delta_width)), img_top = int(max(0.0, face.top() - delta_up));
    int img_right = int(min((double)width, face.right() + delta_width)), img_bottom = int(min((double)height, face.bottom() + delta_down));
    image = image(cv::Rect(cv::Point(img_left, img_top), cv::Point(img_right, img_bottom))).clone();
    face = dlib::rectangle(face.left() - img_left, face.top() - img_top, face.right() - img_left, face.bottom() - img_top);
    dlib::rectangle face_expand(img_left, img_top, img_right, img_bottom);
    cv::Point center(int(face_expand.left() + face_expand.right() + 0.5) / 2, int(face_expand.top() + face_expand.bottom() + 0.5) / 2);
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
        face = dlib::rectangle(face.left() - left, face.top(), face.right() - left, face.bottom());
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
        face = dlib::rectangle(face.left(), face.top() - top, face.right(), face.bottom() - top);
        crop_top += top;
        crop_bottom = crop_top + width;
    }
    dlib::rectangle crop_face(crop_left, crop_top, crop_right, crop_bottom);
    return crop_face;
}

cv::Mat faceparser(const cv::Mat image)
{
    ncnn::Net net;
    net.load_param("assert/faceparser-sim-opt.param");
    net.load_model("assert/faceparser-sim-opt.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, 512, 512);
    const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
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

std::vector<std::vector<int>> landmark(const cv::Mat image, dlib::rectangle face)
{
    dlib::matrix<dlib::rgb_pixel> dlib_image;
    dlib::assign_image(dlib_image, dlib::cv_image<dlib::rgb_pixel>(image));
    dlib::full_object_detection shape = predictor(dlib_image, face);

    std::vector<std::vector<int>> lms(shape.num_parts());
    for (int i = 0; i < shape.num_parts(); i++) {
        lms[i] = { (int)std::round(shape.part(i).y() * 256.0 / image.cols), (int)std::round(shape.part(i).x() * 256.0 / image.cols) };
    }

    return lms;
}

void copy_area(cv::Mat& tar, cv::Mat& src, std::vector<std::vector<int>> lms)
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

void process(cv::Mat mask, std::vector<std::vector<int>> lms, std::vector<cv::Mat>& mask_aug, std::vector<std::vector<cv::Mat>>& diff_re)
{
    cv::Mat ys(cv::Size(256, 256), CV_64FC1);
    for (int i = 0; i < ys.rows; i++)
        for (int j = 0; j < ys.cols; j++)
            ys.at<double>(i,j) = i;
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

    // mask_lip : Mat (256,256) CV_32FC1
    // mask_face : Mat (256,256) CV_32FC1
    // mask_eyes : Mat (256,256) CV_32FC1
    //std::vector<cv::Mat> mask_aug = { mask_lip.clone(), mask_face.clone(), mask_eyes.clone() };
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
    //std::vector<std::vector<cv::Mat>> diff_re(std::vector<std::vector<cv::Mat>>(3, std::vector<cv::Mat>(136)));
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

void preprocess(const cv::Mat in_image, cv::Mat& image, std::vector<cv::Mat>& mask_aug, std::vector<std::vector<cv::Mat>>& diff_re, dlib::rectangle& crop_face)
{
    image = in_image.clone();

    std::vector<dlib::rectangle> faces = detect(image);

    dlib::rectangle face_on_image = faces[0], face = faces[0];

    crop_face = crop(image, face, 0.6 / 0.85, 0.2 / 0.85, 0.2 / 0.85);

    cv::Mat mask = faceparser(image);

    cv::resize(mask, mask, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);

    std::vector<std::vector<int>> lms = landmark(image, face);

    process(mask, lms, mask_aug, diff_re);

    cv::resize(image, image, cv::Size(256, 256));
}

ncnn::Mat from_64_64_136(std::vector<cv::Mat> in)
{
    ncnn::Mat out(64, 64, 136);
    for (int i = 0; i < 136; i++) {
        double* data = (double*)in[i].data;
        float* pt = out.channel(i);
        for (int j = 0; j < 64 * 64; j++) {
            *pt = *data;
            //*pt = 1.f;
            pt++;
            data++;
        }
    }
    return out;
}

ncnn::Mat from_256_256_1(cv::Mat in)
{
    ncnn::Mat out(256, 256, 1);
    double* data = (double*)in.data;
    float* pt = out.channel(0);
    for (int j = 0; j < 256 * 256; j++) {
        *pt = *data;
        //*pt = 1.f;
        pt++;
        data++;
    }
    return out;
}

void show(ncnn::Mat in)
{
    cout << "(" << in.c << "," << in.h << "," << in.w << ")" << endl;
}

void postprocess(cv::Mat source, cv::Mat& result)
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


void softmax(float* src, float*dst)
{
    float alpha = *std::max_element(src, src+4096);
    float denominator = 0;

    for (int i = 0; i < 4096; ++i) {
        dst[i] = std::exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < 4096; ++i) {
        dst[i] /= denominator;
    }
}

int main()
{
    dlib::deserialize("assert/lms.dat") >> predictor;
    

    cv::Mat source = cv::imread("assert/source.png");
    cv::cvtColor(source, source, cv::COLOR_BGR2RGB);
    cv::Mat reference = cv::imread("assert/reference.png");
    cv::cvtColor(reference, reference, cv::COLOR_BGR2RGB);

    cv::Mat real_A, real_B;
    std::vector<cv::Mat> mask_A, mask_B;
    std::vector<std::vector<cv::Mat>> diff_A, diff_B;
    dlib::rectangle crop_face_A, crop_face_B;
    preprocess(source, real_A, mask_A, diff_A, crop_face_A);
    preprocess(reference, real_B, mask_B, diff_B, crop_face_B);

    
    ncnn::Net forward1;
    forward1.load_param("assert/forward1-sim-opt.param");
    forward1.load_model("assert/forward1-sim-opt.bin");

    const float mean_vals[3] = { 0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f };
    const float norm_vals[3] = { 1 / 0.5f / 255.f, 1 / 0.5f / 255.f, 1 / 0.5f / 255.f };
    ncnn::Mat ncnn_real_A = ncnn::Mat::from_pixels(real_A.data, ncnn::Mat::PIXEL_RGB, real_A.cols, real_A.rows);
    ncnn::Mat ncnn_real_B = ncnn::Mat::from_pixels(real_B.data, ncnn::Mat::PIXEL_RGB, real_B.cols, real_B.rows);
    ncnn_real_A.substract_mean_normalize(mean_vals, norm_vals);
    ncnn_real_B.substract_mean_normalize(mean_vals, norm_vals);
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

    ncnn::Extractor ex1 = forward1.create_extractor();
    ex1.set_num_threads(8);
    ex1.set_light_mode(true);
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

    ncnn::Net forward2;
    forward2.load_param("assert/forward2-sim-opt.param");
    forward2.load_model("assert/forward2-sim-opt.bin");

    ncnn::Extractor ex2 = forward2.create_extractor();
    ex2.set_num_threads(8);
    ex2.set_light_mode(true);
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

    cv::Mat source_crop = source(cv::Rect(cv::Point(crop_face_A.left(), crop_face_A.top()), cv::Point(crop_face_A.right(), crop_face_A.bottom()))).clone();

    postprocess(source_crop, fake_A);

    cv::cvtColor(fake_A, fake_A, cv::COLOR_RGB2BGR);
    cv::imwrite("result.png", fake_A);
    

    return 0;
}

