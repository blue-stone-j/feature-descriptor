
#pragma

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

class Display
{
 public:
  Display() {}
  void mosaic_pyramid(const std::vector<std::vector<cv::Mat>> &pyramid, cv::Mat &pyramid_image, int nOctaceLayers, std::string str);

  void write_mosaic_pyramid(const std::vector<std::vector<cv::Mat>> &gauss_pyr_1, const std::vector<std::vector<cv::Mat>> &dog_pyr_1,
                            const std::vector<std::vector<cv::Mat>> &gauss_pyr_2, const std::vector<std::vector<cv::Mat>> &dog_pyr_2, int nOctaveLayers);

  // 没用到，后文没有对其进行定义
  void write_keys_image(std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2,
                        const cv::Mat &image_1, const cv::Mat &image_2, cv::Mat &image_1_keys, cv::Mat &image_2_keys, bool double_size);
};