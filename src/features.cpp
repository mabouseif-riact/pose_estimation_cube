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

    // https://github.com/PointCloudLibrary/pcl/blob/master/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/feature_wrapper/global/ourcvfh_estimator.h
    //
    // You can use the "getTransforms()" function to get the transformations aligning the cloud to the corresponding SGURF.

    pcl::PointCloud<pcl::VFHSignature308>::Ptr ourcvfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

    // OUR-CVFH estimation object.
    pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setInputCloud(cloud);
    ourcvfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ourcvfh.setSearchMethod(tree);
    // ourcvfh.setClusterTolerance (0.015f); //1.5cm, three times the leaf size
    ourcvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees. // or 0.13f
    ourcvfh.setCurvatureThreshold(1.0); // 0.025f
    ourcvfh.setNormalizeBins(false);
    // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
    // this will decide if additional Reference Frames need to be created, if ambiguous.
    ourcvfh.setAxisRatio(0.8);
    // ourcvfh.setRefineClusters(0.8);

    ourcvfh.compute(*ourcvfhs);


    return ourcvfhs;
}



pcl::PointCloud<pcl::VFHSignature308>::Ptr computeOURCVFH(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                          pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& transforms,
                                                          std::vector<bool>& valid_roll_transforms,
                                                          std::vector<pcl::PointIndices>& cluster_indices,
                                                          std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& centroids)
{

    // https://github.com/PointCloudLibrary/pcl/blob/master/apps/3d_rec_framework/include/pcl/apps/3d_rec_framework/feature_wrapper/global/ourcvfh_estimator.h
    //
    // You can use the "getTransforms()" function to get the transformations aligning the cloud to the corresponding SGURF.

    pcl::PointCloud<pcl::VFHSignature308>::Ptr ourcvfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

    // OUR-CVFH estimation object.
    pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setInputCloud(cloud);
    ourcvfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ourcvfh.setSearchMethod(tree);
    // ourcvfh.setClusterTolerance (0.015f); //1.5cm, three times the leaf size
    ourcvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees. // or 0.13f
    ourcvfh.setCurvatureThreshold(0.025); // 0.025f // 1.0f
    ourcvfh.setNormalizeBins(false);
    // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
    // this will decide if additional Reference Frames need to be created, if ambiguous.
    ourcvfh.setAxisRatio(0.8);
    // ourcvfh.setRefineClusters(0.8);

    ourcvfh.compute(*ourcvfhs);

    ourcvfh.getCentroidClusters(centroids);
    ourcvfh.getTransforms(transforms);
    ourcvfh.getValidTransformsVec(valid_roll_transforms);
    ourcvfh.getClusterIndices(cluster_indices);

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



pcl::PointCloud<CRH90>::Ptr computeCRH(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                       const pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    // Object for storing the CRH.
    pcl::PointCloud<CRH90>::Ptr histogram(new pcl::PointCloud<CRH90>);

    // CRH estimation object.
    pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, CRH90> crh;
    crh.setInputCloud(cloud);
    crh.setInputNormals(normals);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    crh.setCentroid(centroid);

    // Compute the CRH.
    crh.compute(*histogram);

    return histogram;
}


std::vector<float> alignCRHAngles(pcl::PointCloud<pcl::PointXYZ>::Ptr viewCloud,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud,
                          pcl::PointCloud<pcl::Normal>::Ptr viewNormals,
                          pcl::PointCloud<pcl::Normal>::Ptr clusterNormals)
            {

// Objects for storing the CRHs of both.
    pcl::PointCloud<CRH90>::Ptr viewCRH(new pcl::PointCloud<CRH90>);
    pcl::PointCloud<CRH90>::Ptr clusterCRH(new pcl::PointCloud<CRH90>);
    // Objects for storing the centroids.
    Eigen::Vector4f viewCentroid;
    Eigen::Vector4f clusterCentroid;

    // Note: here you would compute the CRHs and centroids of both clusters.
    // It has been omitted here for simplicity.
    pcl::compute3DCentroid(*viewCloud, viewCentroid);
    pcl::compute3DCentroid(*clusterCloud, clusterCentroid);
    viewCRH = computeCRH(viewCloud, viewNormals);
    clusterCRH = computeCRH(clusterCloud, clusterNormals);


    // CRH alignment object.
    pcl::CRHAlignment<pcl::PointXYZ, 90> alignment;
    alignment.setInputAndTargetView(clusterCloud, viewCloud);
    // CRHAlignment works with Vector3f, not Vector4f.
    Eigen::Vector3f viewCentroid3f(viewCentroid[0], viewCentroid[1], viewCentroid[2]);
    Eigen::Vector3f clusterCentroid3f(clusterCentroid[0], clusterCentroid[1], clusterCentroid[2]);
    alignment.setInputAndTargetCentroids(clusterCentroid3f, viewCentroid3f);

    // Compute the roll angle(s).
    std::vector<float> angles;
    alignment.computeRollAngle(*clusterCRH, *viewCRH, angles);

    if (angles.size() > 0)
    {
        std::cout << "Number of angles where the histograms correlate: " << angles.size() << std::endl;
        std::cout << "List of angles where the histograms correlate:" << std::endl;

        for (int i = 0; i < angles.size(); i++)
        {
            std::cout << "\t" << angles.at(i) << " degrees." << std::endl;
        }

        return angles;
    }

    exit(EXIT_FAILURE);
}


std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
alignCRHTransforms(pcl::PointCloud<pcl::PointXYZ>::Ptr viewCloud,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud,
                  pcl::PointCloud<pcl::Normal>::Ptr viewNormals,
                  pcl::PointCloud<pcl::Normal>::Ptr clusterNormals)
    {

    // Objects for storing the CRHs of both.
    pcl::PointCloud<CRH90>::Ptr viewCRH(new pcl::PointCloud<CRH90>);
    pcl::PointCloud<CRH90>::Ptr clusterCRH(new pcl::PointCloud<CRH90>);
    // Objects for storing the centroids.
    Eigen::Vector4f viewCentroid;
    Eigen::Vector4f clusterCentroid;

    // Note: here you would compute the CRHs and centroids of both clusters.
    // It has been omitted here for simplicity.
    pcl::compute3DCentroid(*viewCloud, viewCentroid);
    pcl::compute3DCentroid(*clusterCloud, clusterCentroid);
    viewCRH = computeCRH(viewCloud, viewNormals);
    clusterCRH = computeCRH(clusterCloud, clusterNormals);


    // CRH alignment object.
    pcl::CRHAlignment<pcl::PointXYZ, 90> alignment;
    alignment.setInputAndTargetView(viewCloud, clusterCloud);
    // CRHAlignment works with Vector3f, not Vector4f.
    Eigen::Vector3f viewCentroid3f(viewCentroid[0], viewCentroid[1], viewCentroid[2]);
    Eigen::Vector3f clusterCentroid3f(clusterCentroid[0], clusterCentroid[1], clusterCentroid[2]);
    alignment.setInputAndTargetCentroids(viewCentroid3f, clusterCentroid3f);

    // Compute the roll angle(s).
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transforms;
    alignment.align(*viewCRH, *clusterCRH);
    alignment.getTransforms(transforms);

    if (transforms.size() > 0)
    {
        std::cout << "Number of transforms where the histograms correlate: " << transforms.size() << std::endl;

    }
    else
    {
        std::cout << std::endl;
        std::cout << "***************************************************************************" << std::endl;
        std::cout << "*                                                                         *" << std::endl;
        std::cout << "*             CRHAlignment failure. Exit..                                *" << std::endl;
        std::cout << "*                                                                         *" << std::endl;
        std::cout << "***************************************************************************" << std::endl << std::endl;

        Eigen::Matrix4f identity_transform = Eigen::Matrix4f::Identity(4, 4);
        transforms.push_back(identity_transform);
        // exit(EXIT_FAILURE);
    }

    return transforms;
}
