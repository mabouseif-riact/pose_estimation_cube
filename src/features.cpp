#include "features.h"




pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod (tree);
    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch (0.03);

    // Compute the features
    fpfh.compute (*fpfhs);

    return fpfhs;

}



pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, bool flip, double radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

    // if (viewpoint)
    //     ne.setViewPoint(0, 0, 0);

    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    ne.setSearchMethod(tree);

    ne.setRadiusSearch(radius);

    ne.compute(*normals);

    if (flip)
        for (auto &normal: *normals)
        {
            normal.normal_x *=-1;
            normal.normal_y *=-1;
            normal.normal_z *=-1;
        }

    return normals;
}


pcl::PointCloud<pcl::VFHSignature308>::Ptr computeVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

    // Create the VFH estimation class, and pass the input dataset+normals to it
    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    vfh.setSearchMethod (tree);

    // Compute the features
    vfh.compute (*vfhs);

    return vfhs;
}


pcl::PointCloud<pcl::VFHSignature308>::Ptr computeCVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{


    // You can further customize the segmentation step with "setClusterTolerance()" (to set the maximum 
    // Euclidean distance between points in the same cluster) and "setMinPoints()". The size of the 
    // output will be equal to the number of regions the object was divided in. Also, check the 
    // functions "getCentroidClusters()" and "getCentroidNormalClusters()", you can use them to 
    // get information about the centroids used to compute the different CVFH descriptors. 

    pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

    // Create the VFH estimation class, and pass the input dataset+normals to it
    pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh;
    cvfh.setInputCloud(cloud);
    cvfh.setInputNormals(normals);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    cvfh.setSearchMethod (tree);
    // Set the maximum allowable deviation of the normals,
    // for the region segmentation step.
    cvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
    // Set the curvature threshold (maximum disparity between curvatures),
    // for the region segmentation step.
    cvfh.setCurvatureThreshold(1.0);
    // Set to true to normalize the bins of the resulting histogram,
    // using the total number of points. Note: enabling it will make CVFH
    // invariant to scale just like VFH, but the authors encourage the opposite.
    cvfh.setNormalizeBins(false);

    // Compute the features
    cvfh.compute (*cvfhs);

    return cvfhs;
}


pcl::PointCloud<pcl::VFHSignature308>::Ptr computeOURCVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                       pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    // You can use the "getTransforms()" function to get the transformations aligning the cloud to the corresponding SGURF. 
    
    pcl::PointCloud<pcl::VFHSignature308>::Ptr ourcvfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

    // OUR-CVFH estimation object.
    pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setInputCloud(cloud);
    ourcvfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ourcvfh.setSearchMethod(tree);
    ourcvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
    ourcvfh.setCurvatureThreshold(1.0);
    ourcvfh.setNormalizeBins(false);
    // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
    // this will decide if additional Reference Frames need to be created, if ambiguous.
    ourcvfh.setAxisRatio(0.8);

    ourcvfh.compute(*ourcvfhs);

    return ourcvfhs;
}

pcl::PointCloud<pcl::Histogram <135> >::Ptr computeROPS(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    float support_radius = 0.005f;
    unsigned int number_of_partition_bins = 20;
    unsigned int number_of_rotations = 3;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method (new pcl::search::KdTree<pcl::PointXYZ>);
    search_method->setInputCloud (cloud);

    pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram<135>> feature_estimator;
    feature_estimator.setSearchMethod (search_method);
    feature_estimator.setSearchSurface (cloud);
    feature_estimator.setInputCloud (cloud);
    // feature_estimator.setIndices (indices);
    // feature_estimator.setTriangles (triangles);
    feature_estimator.setRadiusSearch (support_radius);
    feature_estimator.setNumberOfPartitionBins (number_of_partition_bins);
    feature_estimator.setNumberOfRotations (number_of_rotations);
    feature_estimator.setSupportRadius (support_radius);

    pcl::PointCloud<pcl::Histogram<135>>::Ptr histograms (new pcl::PointCloud <pcl::Histogram<135>> ());
    feature_estimator.compute (*histograms);

    return histograms;
}



