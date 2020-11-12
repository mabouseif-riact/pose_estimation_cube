#include <pcl/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/feature_wrapper/local/fpfh_local_estimator.h>
#include <pcl/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/feature_wrapper/local/shot_local_estimator.h>
#include <pcl/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/feature_wrapper/local/shot_local_estimator_omp.h>
#include <pcl/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/pc_source/mesh_source.h>
#include <pcl/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/pipeline/local_recognizer.h>
#include <pcl/common/transforms.h> // for transformPointCloud
#include <pcl/console/parse.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/recognition/hv/hv_papazov.h>
#include <pcl/visualization/pcl_visualizer.h>





template<template <class> class DistT, typename PointT, typename FeatureT>
void recognizeAndVisualize(typename pcl::rec_3d_framework::LocalRecognitionPipeline<DistT, PointT, FeatureT>& local,
                           std::string& scenes_dir,
                           int scene = -1,
                           bool single_model = false)
{

}
