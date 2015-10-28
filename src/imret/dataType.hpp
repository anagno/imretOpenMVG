#ifndef DATATYPE_HPP_
#define DATATYPE_HPP_


#include <string>
#include <memory>

#include "third_party/eigen/Eigen/SparseCore"
#include "third_party/eigen/Eigen/Core"
#include "flann/flann.hpp"

typedef float Scalar;
typedef flann::L2<Scalar> Metric;
typedef Metric::ResultType DistanceType;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfr;
typedef Eigen::Triplet<float> Tri;
//typedef openMVG::features::Descriptor<unsigned char,128>::bin_type  DescTbin;

namespace imretDataTypes {

        struct ENV{
                std::string img_path;
                std::string sim_path;
                std::string imfmt;
        };
        
        struct opt{
                unsigned int vvrepeats;
                unsigned int vviter;
                double vvsizeratio;
                int  vvmaxsize;
                const char* sadr_vv;
        };

        struct vv_struct{
                MatrixXfr CX;
                std::vector<float> sses;
                std::vector<int> CN;
                std::vector<int> assgn;
        };
        
        struct vv_flann{
                std::unique_ptr<flann::Index<Metric> > fl_idx;
                int fl_params;
        };
        struct data_qr{
                std::vector<int> assgn;
                std::vector<int> vw2im;

        };

        struct data_vv{
                std::vector<int> CN;
                int N;
        };
        template<class T> 
        struct index_cmp {
                index_cmp(const T arr) : arr(arr) {}
                bool operator()(const size_t a, const size_t b) const{ 
                        return arr[a] < arr[b];
                }
                const T arr;
        };
}


#endif



