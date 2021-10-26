#pragma once
#include "net.h"
#include "gpu.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#define NMS_UNION 1
#define NMS_MIN  2

typedef struct FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float area;

	float landmarks[10];
};

class LFFD {
public:
	LFFD();
	~LFFD();

	int detect(ncnn::Mat& img, std::vector<FaceInfo>& face_lis,int resize_h,int resize_w,
		float score_threshold = 0.6, float nms_threshold = 0.4, int top_k = 10000,
		std::vector<int> skip_scale_branch_list = {});

private:
	void generateBBox(std::vector<FaceInfo>& collection, ncnn::Mat score_map, ncnn::Mat box_map, float score_threshold,
		int fea_w, int fea_h, int cols, int rows, int scale_id);
	void get_topk_bbox(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, int topk);
	void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
		float threshold, int type = NMS_MIN);
private:
	ncnn::Net lffd;

	int num_thread;
	int num_output_scales;
	int image_w;
	int image_h;

	std::string param_file_name;
	std::string bin_file_name;

	std::vector<float> receptive_field_list;
	std::vector<float> receptive_field_stride;
	std::vector<float> bbox_small_list;
	std::vector<float> bbox_large_list;
	std::vector<float> receptive_field_center_start;
	std::vector<float> constant;

	std::vector<std::string> output_blob_names;

	const float mean_vals[3] = { 127.5, 127.5, 127.5 };
	const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
};
