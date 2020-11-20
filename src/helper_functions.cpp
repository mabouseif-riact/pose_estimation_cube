#include "helper_functions.h"



void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field, double min_val, double max_val)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min_val, max_val);
    pass.filter(*cloud);
}


void scaleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale)
{
    // Something wrong here. It should be 3x3
    Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Identity(4, 4);
    affine_mat *= scale;
    pcl::transformPointCloud(*cloud, *cloud, affine_mat);
}


void moveCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const char axis, double dist)
{
    // Something wrong here. It should be 3x3
    Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Identity(4, 4);
    switch(axis)
    {
        case 'x':
            affine_mat(3, 0) = dist;
            break;
        case 'y':
            affine_mat(3, 1) = dist;
            break;
        case 'z':
            affine_mat(3, 2) = dist;
            break;
        default:
            PCL_ERROR("INVALID AXIS!");
            break;
    }

    pcl::transformPointCloud(*cloud, *cloud, affine_mat);
}


void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (200);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud);
}



std::vector<pcl::PointIndices> clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::vector<pcl::PointIndices> cluster_indices;
    {

    pcl::ScopeTime t("Clustering");

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    }

    return cluster_indices;
}




int PCDIndex(std::string path)
{
    // std::string path = entry.path().string();
    std::string delimiter = ".";
    std::string token = path.substr(0, path.find(delimiter));
    std::string num_str = "";
    for (std::string::reverse_iterator rev_it = token.rbegin(); rev_it != token.rbegin() + 2; ++rev_it)
    {
        if ((int)*rev_it >= 48 && (int)*rev_it <= 57)
            num_str += *rev_it;
    }
    std::reverse(num_str.begin(), num_str.end());

    return std::stoi(num_str);
}



std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> openData(std::string fileToOpen)
{

    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format



    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    std::vector<double> matrixEntries;

    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);

    // this variable is used to store the row of the matrix that contains commas
    std::string matrixRowString;

    // this variable is used to store the matrix entry;
    std::string matrixEntry;

    // this variable is used to track the number of rows
    int matrixRowNumber = 0;

    double a, b, c, d;
    Eigen::Matrix4f temp_mat;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new;


    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::istringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

        matrixRowStringStream >> a >> b >> c >> d;
        temp_mat.row(matrixRowNumber) << a, b, c, d;

        matrixRowNumber++; //update the column

        if (((matrixRowNumber+1) % 5) == 0)
        {
            matrixRowNumber = 0;
            poses_new.push_back(temp_mat);
        }

    }

    // here we convet the vector variable into the matrix and return the resulting object,
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    // return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);

    return poses_new;
}



std::vector<std::vector<float>> readCRH(std::string file_name)
{
    std::ifstream CRH_file(file_name);
    std::string crh_string;
    float num;
    std::vector<float> temp_hist_vec;
    std::vector<std::vector<float>> hist_vecs;

    std::cout << "\n\n\n" << std::endl;
    while (getline(CRH_file, crh_string)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::istringstream matrixRowStringStream(crh_string); //convert matrixRowString that is a string to a stream variable.
        while (matrixRowStringStream >> num)
            temp_hist_vec.push_back(num);
        hist_vecs.push_back(temp_hist_vec);
        temp_hist_vec.clear();
    }

    return hist_vecs;
}


