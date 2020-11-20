#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/time.h>
#include <pcl/common/common_headers.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>


void ICP(pcl::PointCloud<pcl::PointXYZ>::ConstPtr object_aligned, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster);

pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, double dist_thresh=0.01, int max_iterations=1000);

template <typename featureType> // || TEMPLATE!!
pcl::PointCloud<pcl::PointXYZ>::Ptr RANSACPrerejective(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                           pcl::PointCloud<pcl::PointXYZ>::ConstPtr target,
                                           featureType cloud_features,
                                           featureType target_featries)
{
    // Perform alignment
    pcl::console::print_highlight ("Starting alignment...\n");
    const float leaf = 0.005f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::VFHSignature308> align;
    // pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::Histogram<135>> align;
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> align;
    align.setInputSource (cloud);
    align.setSourceFeatures (cloud_features);
    align.setInputTarget (target);
    align.setTargetFeatures (target_featries);
    align.setMaximumIterations (50000); // Number of RANSAC iterations
    align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness (20); // Number of nearest features to use
    align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance (2.5f * leaf); // Inlier threshold
    align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis
    {
      pcl::ScopeTime t("Alignment");
      align.align (*object_aligned);
    }

    std::cout << "RANSAC Convergence: " << align.hasConverged() << std::endl;

    return object_aligned;
}
