
#include "sift.h"
#include "display.h"
#include "match.h"
#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <stdlib.h>
#include <filesystem>

int main(int argc, char *argv[])
{
  /********************** 1、读入数据 **********************/
  cv::Mat image_1 = cv::imread("../assets/1.pgm", -1);
  cv::Mat image_2 = cv::imread("../assets/2.pgm", -1);

  std::string change_model = "perspective"; // affine为仿射变换，初始为perspective

  std::string folderPath = "../assets";

  double total_count_beg = (double)cv::getTickCount(); // 算法运行总时间开始计时

  Sift sift_1(0, 3, 0.04, 10, 1.6, true); // 类对象

  /********************** 1、参考图像特征点检测和描述 **********************/

  std::vector<std::vector<cv::Mat>> gauss_pyr_1, dog_pyr_1;          // 高斯金字塔和高斯差分金字塔
  std::vector<cv::KeyPoint> keypoints_1;                             // 特征点
  std::vector<std::vector<std::vector<float>>> all_cell_contrasts_1; // 所有尺度层中所有单元格的对比度
  std::vector<std::vector<float>> average_contrast_1;                // 所有尺度层中多有单元格的平均对比度
  std::vector<std::vector<int>> n_cells_1;                           // 所有尺度层，每一尺度层中所有单元格所需特征数
  std::vector<int> num_cell_1;                                       // 当前尺度层中所有单元格中的所需特征数
  std::vector<std::vector<int>> available_n_1;                       // 所有尺度层，每一尺度层中所有单元格可得到特征数
  std::vector<int> available_num_1;                                  // 一个尺度层中所有单元格中可用特征数量
  std::vector<cv::KeyPoint> final_keypoints1;                        // 第一次筛选结果
  std::vector<cv::KeyPoint> Final_keypoints1;                        // 第二次筛选结果
  std::vector<cv::KeyPoint> Final_Keypoints1;                        // 第三次筛选结果

  cv::Mat descriptors_1; // 特征描述子

  double detect_1 = (double)cv::getTickCount(); // 特征点检测运行时间开始计时

  sift_1.detect(image_1, gauss_pyr_1, dog_pyr_1, keypoints_1, all_cell_contrasts_1,
                average_contrast_1, n_cells_1, num_cell_1, available_n_1, available_num_1,
                final_keypoints1, Final_keypoints1, Final_Keypoints1); // 特征点检测

  std::cout << "参考图像检测出的总特征数 =" << keypoints_1.size() << std::endl;

  // cv::getTickFrequency() 返回值为一秒的计时周期数，二者比值为特征点检测时间`
  double detect_time_1 = ((double)cv::getTickCount() - detect_1) / cv::getTickFrequency();

  std::cout << "参考图像特征点检测时间是： " << detect_time_1 * 1000 << "ms" << std::endl;

  double comput_1 = (double)cv::getTickCount();

  std::vector<cv::Mat> sar_harris_fun_1;
  std::vector<cv::Mat> amplit_1;
  std::vector<cv::Mat> orient_1;

  sift_1.comput_des(gauss_pyr_1, keypoints_1, amplit_1, orient_1, descriptors_1);

  double comput_time_1 = ((double)cv::getTickCount() - comput_1) / cv::getTickFrequency();

  std::cout << "参考图像特征点描述时间是： " << comput_time_1 * 1000 << "ms" << std::endl;

  /********************** 1、待配准图像特征点检测和描述 **********************/

  std::vector<std::vector<cv::Mat>> gauss_pyr_2, dog_pyr_2;
  std::vector<cv::KeyPoint> keypoints_2;
  std::vector<std::vector<std::vector<float>>> all_cell_contrasts_2; // 所有尺度层中所有单元格的对比度
  std::vector<std::vector<float>> average_contrast_2;                // 所有尺度层中多有单元格的平均对比度
  std::vector<std::vector<int>> n_cells_2;                           // 所有尺度层，每一尺度层中所有单元格所需特征数
  std::vector<int> num_cell_2;                                       // 当前尺度层中所有单元格中的所需特征数
  std::vector<std::vector<int>> available_n_2;                       // 所有尺度层，每一尺度层中的一个单元格可得到特征数
  std::vector<int> available_num_2;                                  // 一个尺度层中所有单元格中可用特征数量
  std::vector<cv::KeyPoint> final_keypoints2;                        // 第一次筛选结果
  std::vector<cv::KeyPoint> Final_keypoints2;                        // 第二次筛选结果
  std::vector<cv::KeyPoint> Final_Keypoints2;                        // 第三次筛选结果

  cv::Mat descriptors_2;

  double detect_2 = (double)cv::getTickCount();

  sift_1.detect(image_2, gauss_pyr_2, dog_pyr_2, keypoints_2, all_cell_contrasts_2,
                average_contrast_2, n_cells_2, num_cell_2, available_n_2, available_num_2, final_keypoints2, Final_keypoints2, Final_Keypoints2);

  std::cout << "待配准图像检测出的总特征数 =" << keypoints_2.size() << std::endl;

  double detect_time_2 = ((double)cv::getTickCount() - detect_2) / cv::getTickFrequency();

  std::cout << "待配准图像特征点检测时间是： " << detect_time_2 * 1000 << "ms" << std::endl;

  double comput_2 = (double)cv::getTickCount();

  std::vector<cv::Mat> sar_harris_fun_2;
  std::vector<cv::Mat> amplit_2;
  std::vector<cv::Mat> orient_2;

  sift_1.comput_des(gauss_pyr_2, keypoints_2, amplit_2, orient_2, descriptors_2);

  double comput_time_2 = ((double)cv::getTickCount() - comput_2) / cv::getTickFrequency();

  std::cout << "待配准特征点描述时间是： " << comput_time_2 * 1000 << "ms" << std::endl;

  /********************** 1、最近邻与次近邻距离比匹配，两幅影像进行配准 **********************/

  SiftMatch mymatch;

  double match_time = (double)cv::getTickCount(); // 影像配准计时开始

  // knnMatch函数是DescriptorMatcher类的成员函数，FlannBasedMatcher是DescriptorMatcher的子类
  cv::Ptr<cv::DescriptorMatcher> matcher1 = new cv::FlannBasedMatcher;
  cv::Ptr<cv::DescriptorMatcher> matcher2 = new cv::FlannBasedMatcher;

  std::vector<std::vector<cv::DMatch>> dmatchs; // std::vector<cv::DMatch>中存放的是一个描述子可能匹配的候选集
  std::vector<std::vector<cv::DMatch>> dmatch1;
  std::vector<std::vector<cv::DMatch>> dmatch2;

  // calculate initial matches
  /* (查询描述符集合,训练描述符集合,存储匹配结果,指定返回每个查询点的K个最佳匹配)*/
  matcher1->knnMatch(descriptors_1, descriptors_2, dmatchs, 2);

  std::cout << "距离比之前初始匹配点对个数是：" << dmatchs.size() << std::endl;

  cv::Mat matched_lines;                             // 同名点连线
  std::vector<cv::DMatch> init_matchs, right_matchs; // 用于存放正确匹配的点

  // 该函数返回的是空间映射模型参数
  cv::Mat homography = mymatch.match(image_1, image_2, dmatchs, keypoints_1, keypoints_2, change_model, right_matchs, matched_lines, init_matchs);

  double match_time_2 = ((double)cv::getTickCount() - match_time) / cv::getTickFrequency();

  std::cout << "特征点匹配花费时间是： " << match_time_2 * 1000 << "ms" << std::endl;
  std::cout << change_model << "变换矩阵是：" << std::endl;
  std::cout << homography << std::endl;

  /********************** 1、把正确匹配点坐标写入文件中 **********************/

  std::ofstream ofile;
  ofile.open("./position.txt"); // 创建文件
  for (size_t i = 0; i < right_matchs.size(); ++i)
  {
    ofile << keypoints_1[right_matchs[i].queryIdx].pt << "   "
          << keypoints_2[right_matchs[i].trainIdx].pt << std::endl;
  }

  /********************** 1、图像融合 **********************/

  double fusion_beg = (double)cv::getTickCount();

  cv::Mat fusion_image, mosaic_image, regist_image;
  mymatch.image_fusion(image_1, image_2, homography, fusion_image, regist_image);

  imwrite("../image_save/融合后的图像.jpg", fusion_image);

  double fusion_time = ((double)cv::getTickCount() - fusion_beg) / cv::getTickFrequency();
  std::cout << "图像融合花费时间是： " << fusion_time * 1000 << "ms" << std::endl;

  double total_time = ((double)cv::getTickCount() - total_count_beg) / cv::getTickFrequency();
  std::cout << "总花费时间是： " << total_time * 1000 << "ms" << std::endl;

  /********************** 1、生成棋盘图，显示配准结果 **********************/

  // 生成棋盘网格图像
  cv::Mat chessboard_1, chessboard_2, mosaic_images;

  cv::Mat image1 = cv::imread("../image_save/配准后的参考图像.jpg", -1);
  cv::Mat image2 = cv::imread("../image_save/配准后的待配准图像.jpg", -1);

  mymatch.mosaic_map(image1, image2, chessboard_1, chessboard_2, mosaic_images, 50);

  imwrite("../image_save/参考图像棋盘图像.jpg", chessboard_1);
  imwrite("../image_save/待配准图像棋盘图像.jpg", chessboard_2);
  imwrite("../image_save/两幅棋盘图像叠加.jpg", mosaic_images);

  return 0;
}