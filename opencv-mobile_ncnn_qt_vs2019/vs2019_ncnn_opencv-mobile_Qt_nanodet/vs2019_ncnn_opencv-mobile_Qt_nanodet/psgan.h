#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include "net.h"
#include <opencv2/opencv.hpp>
#include "LFFD.h"

using namespace std;

class PSGAN
{
public:
	PSGAN();
	~PSGAN();
public:
	cv::Mat go(const cv::Mat source, const cv::Mat _reference);
private:
    std::vector<pair<cv::Point, cv::Point>> detect(cv::Mat image);
    pair<cv::Point, cv::Point> crop(cv::Mat& image, pair<cv::Point, cv::Point>& face, double up_ratio, double down_ratio, double width_ratio);
    cv::Mat faceparser(const cv::Mat image);
    std::vector<std::vector<int>> landmark(const cv::Mat image, pair<cv::Point, cv::Point> face);
    void copy_area(cv::Mat& tar, cv::Mat& src, std::vector<std::vector<int>> lms);
    void process(cv::Mat mask, std::vector<std::vector<int>> lms, std::vector<cv::Mat>& mask_aug, std::vector<std::vector<cv::Mat>>& diff_re);
    void preprocess(const cv::Mat in_image, cv::Mat& image, std::vector<cv::Mat>& mask_aug, std::vector<std::vector<cv::Mat>>& diff_re, pair<cv::Point, cv::Point>& crop_face);
    ncnn::Mat from_64_64_136(std::vector<cv::Mat> in);
    ncnn::Mat from_256_256_1(cv::Mat in);
    void postprocess(cv::Mat source, cv::Mat& result);
private:
    int num_thread;

    LFFD lffd;
    ncnn::Net FaceParserNet;
    ncnn::Net LandmarkNet;
    ncnn::Net forward1Net;
    ncnn::Net forward2Net;

    const float FaceParserNet_mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float FaceParserNet_norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
    const float LandmarkNet_mean_vals[3] = { 127.f, 127.f, 127.f };
    const float LandmarkNet_norm_vals[3] = { 1.f / 127.f,1.f / 127.f, 1.f / 127.f };
    const float forward1Net_mean_vals[3] = { 0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f };
    const float forward1Net_norm_vals[3] = { 1 / 0.5f / 255.f, 1 / 0.5f / 255.f, 1 / 0.5f / 255.f };
};