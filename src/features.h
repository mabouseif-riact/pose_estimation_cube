#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/crh.h>
#include <pcl/recognition/crh_alignment.h>

typedef pcl::Histogram<90> CRH90;

pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, bool flip, double radius=0.03);

pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals);

pcl::PointCloud<pcl::Histogram<135>>::Ptr computeROPS(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

pcl::PointCloud<pcl::VFHSignature308>::Ptr computeVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals);


pcl::PointCloud<pcl::VFHSignature308>::Ptr computeCVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals);

pcl::PointCloud<pcl::VFHSignature308>::Ptr computeOURCVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals);

pcl::PointCloud<CRH90>::Ptr computeCRH(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, 
                					   const pcl::PointCloud<pcl::Normal>::ConstPtr normals);

void alignCRH(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr viewCloud, 
              const pcl::PointCloud<pcl::PointXYZ>::ConstPtr clusterCloud, 
              const pcl::PointCloud<pcl::Normal>::ConstPtr viewNormals, 
              const pcl::PointCloud<pcl::Normal>::ConstPtr clusterNormals);