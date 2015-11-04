
#ifndef IMRETFUNCS_HPP_
#define IMRETFUNCS_HPP_


#if !defined(SIFTGPU_STATIC) && !defined(SIFTGPU_DLL_RUNTIME)
// SIFTGPU_STATIC comes from compiler
#define SIFTGPU_DLL_RUNTIME
// Load at runtime if the above macro defined
// comment the macro above to use static linking
#endif
////////////////////////////////////////////////////////////////////////////
// define REMOTE_SIFTGPU to run computation in multi-process (Or remote) mode
// in order to run on a remote machine, you need to start the server manually
// This mode allows you use Multi-GPUs by creating multiple servers
// #define REMOTE_SIFTGPU
// #define REMOTE_SERVER        NULL
// #define REMOTE_SERVER_PORT   7777


///////////////////////////////////////////////////////////////////////////
#define DEBUG_SIFTGPU  //define this to use the debug version in windows

#ifdef _WIN32
#ifdef SIFTGPU_DLL_RUNTIME
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FREE_MYLIB FreeLibrary
#define GET_MYPROC GetProcAddress
#else
//define this to get dll import definition for win32
#define SIFTGPU_DLL
#ifdef _DEBUG
#pragma comment(lib, "../thirdparty/SiftGPU/lib/siftgpu_d.lib")
#else
#pragma comment(lib, "../thirdparty/SiftGPU/lib/siftgpu.lib")
#endif
#endif
#else
#ifdef SIFTGPU_DLL_RUNTIME
#include <dlfcn.h>
#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#endif
#endif

#include "dataType.hpp"
#include "openMVG/matching/matcher_kdtree_flann.hpp"
#include "openMVG/features/features.hpp"
#include "openMVG_dependencies/nonFree/SIFT_describer.hpp"
#include "openMVG/image/image.hpp"
#include "openMVG/numeric/numeric.h"
#include "openMVG/multiview/solver_homography_kernel.hpp"
#include "openMVG/multiview/conditioning.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansac.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansacKernelAdaptator.hpp"

#include "third_party/vectorGraphics/svgDrawer.hpp"
//#include "third_party/flann/src/cpp/flann/flann.hpp"
#include "flann/flann.hpp" //use flann 1.8
#include "third_party/eigen/Eigen/Core"
#include "third_party/eigen/Eigen/StdVector"
#include "third_party/eigen/unsupported/Eigen/MatrixFunctions"
#include "third_party/eigen/Eigen/SparseCore"
#include "third_party/eigen/unsupported/Eigen/SparseExtra"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "../thirdparty/SiftGPU/src/SiftGPU/SiftGPU.h"
#include "../thirdparty/USAC/src/estimators/HomogEstimator.h"

#include <vector>
#include <utility>
#include <memory>
#include <cassert>
#include <ctime>
#include <sys/time.h>
#include <math.h>
#include <algorithm>
#include <functional>
#include <dlfcn.h>
#include <fstream>
#include <iterator>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>



#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym



///////////////////////////////////////////////////////////global varaibles///////////////////////////////////////////////////////
struct timeval starttime, endtime;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool compareNat(const std::string& a, const std::string& b);
bool is_not_digit(char c);
bool numeric_string_compare(const std::string& s1, const std::string& s2);
void get_labels_db(const std::string& labelPath, std::vector<std::vector<double>>& labels);
void getGazes(const std::string& gazePath, std::vector<std::vector<double> >& gazes);
void get_gps_qr(const std::string qr_label_path,const std::string gps_qr_path, std::vector<std::vector<double> >& labels );
double  getDisPnt2Vert(std::pair<double, double> pnt, std::pair<double, double> vert1, std::pair<double, double> vert2);
void getGPSsimil(const std::string& gps_db_path, const std::string& gps_qr_path);
void getDBImID(const std::string& gps_proc_path, const std::string& label_db_path, std::vector<std::vector<double> >& gps_assgn);
int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy);

int compute1(const imretDataTypes::opt& pOpt, std::vector<std::string>& paths, int topn, float match_thr );
int compute2(const imretDataTypes::opt& pOpt, std::vector<std::string>& paths, int topn, float match_thr );
int compute3(const imretDataTypes::opt& pOpt, std::vector<std::string>& paths, int topn, float match_thr );
int compute(const int alg_idx, const std::string& GPS_path, const std::string& gaze_path, const std::string& qr_img_path);
int printout();

//////////////////////////////////////////////////class imretFuncs//////////////////////////////////////////////////////////////////

class imretFuncs {

private:
    std::string _rec_path;
    imretDataTypes::ENV _env;
    void * hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
    template< class T >
    void reorder(std::vector<T> & unordered, std::vector<size_t> const & index_map, std::vector<T> & ordered);
    template<class Derived>
    void flann_approx_kmeans(const Eigen::MatrixBase<Derived>& X,   Eigen::MatrixBase<Derived> & CX, int nclus,  std::vector<float>& sses, std::vector<int>& CN,  std::vector<int>& assgn);

public:
    //constructors
    imretFuncs();
    imretFuncs(const std::string& rec_path);
    imretFuncs(const std::string& rec_path, imretDataTypes::ENV env);
    imretFuncs(const std::string& rec_path, imretDataTypes::ENV env, const std::string& imfmt);
    void getEnv(imretDataTypes::ENV& env) const;
    //functions
    void getFileList(std::vector<std::string>& fileList) const;
    //        void simprep(std::vector<std::string>& fileLists) ;
    int simprep_gpu(std::vector<std::string>& fileLists);
    template<class Derived>
    void get_descs_for_vv(int& N,Eigen::MatrixBase<Derived>& desc2vv) ;
    void create_vv(const imretDataTypes::opt& opt);
    void simquant( std::vector<std::string>& img_list, const imretDataTypes::opt& opt,  std::vector<int>& assgn, std::vector<int>& vw2im);
    //template<class Derived>
    //        void simget(const Eigen::SparseMatrixBase<Derived>& sim);
    template<class Derived>
    void convert2flann(const Eigen::MatrixBase<Derived>& mat_eigen,flann::Matrix<Scalar>& mat_flann);
    template<class Derived>
    void tfidf_from_vvQR_fast(const imretDataTypes::data_qr& qr_tfidf, const imretDataTypes::data_vv& vv_tfidf, Eigen::SparseMatrixBase<Derived>& tfidf);
    void sift2dvw(const std::string& dvw_fname, const imretDataTypes::vv_flann& vv, std::vector<int>& assgnimg_perimg);

    void load_ftrs_to_sim( std::vector<int>& assgn, std::vector<int>& vw2im, std::vector<int>& CN, int& N);
    template<class Derived>
    void load_descs(const std::string& dvw_fname_sift,Eigen::MatrixBase<Derived>& descs_mat);
    template<class Derived>
    void load_feats(const std::string& dvw_fname_sift, Eigen::MatrixBase<Derived>& feats_mat);
    void get_sim_path(std::string& sim_path);
    template <class T>
    void sort(std::vector<T> &unsorted,std::vector<T> &sorted,std::vector<size_t> &index_map);

    //destructor
    ~imretFuncs();

};



/////////////////////////////////////class implementations////////////////////////////////////////////////////////////////////

imretFuncs::imretFuncs(){
    this->_rec_path = stlplus::folder_to_path ("../workspace");
    this->_env.img_path = this->_rec_path + "dataset/";
    this->_env.sim_path = this->_rec_path + "sim/";
    this->_env.imfmt = "png";
}

imretFuncs::imretFuncs(const std::string& rec_path){
    this->_rec_path = rec_path;
    this->_env.img_path = this->_rec_path  + "data/";
    this->_env.sim_path = this->_rec_path + "sim/";
    this->_env.imfmt = "png";
}

imretFuncs::imretFuncs(const std::string& rec_path, imretDataTypes::ENV env){
    this->_rec_path = rec_path;
    this->_env.img_path = env.img_path;
    this->_env.sim_path = env.sim_path;
    this->_env.imfmt = "png";
}

imretFuncs::imretFuncs(const std::string& rec_path, imretDataTypes::ENV env, const std::string& imfmt){
    this->_rec_path = rec_path;
    this->_env.img_path = env.img_path;
    this->_env.sim_path = env.sim_path;
    this->_env.imfmt = imfmt;
}

void imretFuncs::getFileList(std::vector<std::string>& fileList) const{
    if( stlplus::file_exists(_env.img_path + ".DS_Store")){
        stlplus::file_delete(_env.img_path + ".DS_Store");
    }

    if(!stlplus::folder_empty(_env.img_path)){
        fileList =  stlplus::folder_files (_env.img_path);
        std::sort(fileList.begin(), fileList.end(), compareNat);
        // std::copy(fileList.begin(), fileList.end(), std::ostream_iterator<std::string>(std::cout, " "));
        // std::cout << std::endl;

    }else{

        throw std::invalid_argument ("empty folder path" + _env.img_path);
    }

}

/*
void imretFuncs::simprep( std::vector<std::string>& fileList) {
        getFileList(fileList);
        int i;
        for(i = 0; i < fileList.size(); ++i){
                //define the sift file name
                std::string featName = _env.sim_path+"dvw_"+ stlplus::basename_part( fileList[i]) + ".feat";
                std::string descName = _env.sim_path+"dvw_"+ stlplus::basename_part( fileList[i]) + ".desc";
                if((!stlplus::file_exists(featName)) && (!stlplus::file_exists(descName))){
                        //load image
                        openMVG::image::Image<unsigned char> grayImage;
                        std::string imagePath =  _env.img_path + fileList[i];
                        // std::cout << imagePath << std::endl;
                        openMVG::image::ReadImage( imagePath.c_str(), &grayImage);

                        //define image_describer and find out the descriptors with vlfeat
                        std::unique_ptr<openMVG::features::Image_describer> image_describer(new openMVG::features::SIFT_Image_describer);
                        std::unique_ptr<openMVG::features::Regions>  regions_ptr;
                        image_describer->Describe(grayImage, regions_ptr);

                        //cast the descriptors into sift
                        const openMVG::features::SIFT_Regions* regions = dynamic_cast<openMVG::features::SIFT_Regions*>(regions_ptr.get());
                        //const openMVG::features::PointFeatures feats = regions_raw->GetRegionsPositions();

                        //store the descriptors

                        regions->Save(featName, descName);
                        // std::ofstream os (featName, std::ofstream::out);
                        // //write number of feats into file
                        // os << feats.size()<<std::endl;
                        // //write feats to file
                        // for (size_t i=0; i < feats.size(); ++i )  {
                        //         const openMVG::features::SIOPointFeature point = regions->Features()[i];
                        //         os<< i<< "\t" <<point.x() <<"\t"<<point.y() << "\t" << point.scale() << "\t"<< point.orientation()<<std::endl;

                        // }
                }
        }
}

*/

int imretFuncs::simprep_gpu( std::vector<std::string>& fileList){


    if(hsiftgpu == NULL) return 0;

    SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
    pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");

    //with cuda, verbose and write results into binary
    //        char * argv[] = {"-fo", "-1",  "-v", "1","-b", "1","-cuda"};
    //with cuda and write results into binary; no verbose
    char * argv[] = {"-fo", "-1","-v", "0","-b", "1","-cuda"};
    //with cuda and verbose, write results into ascii
    //        char * argv[] = {"-fo", "-1",  "-v", "1","-cuda"};
    //with GLSL
    //char * argv[] = {"-fo", "-1",  "-v", "1","-b", "1"}
    int argc = sizeof(argv)/sizeof(char*);

    int i;

    for(i = 0; i < fileList.size(); ++i){

        SiftGPU* sift = pCreateNewSiftGPU(1);
        sift->ParseParam(argc, argv);
        if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

        std::string siftName = _env.sim_path+"dvw_"+ stlplus::basename_part( fileList[i]) + ".sift";
        std::string imagePath =  _env.img_path + fileList[i];
        //                std::cout << siftName << std::endl;
        if(sift->RunSIFT(imagePath.c_str())){
            sift->SaveSIFT(siftName.c_str()); //Note that saving ASCII format is slow

            // vector<float> descriptors(1);
            // vector<SiftGPU::SiftKeypoint> keys(1);

            // //get feature count
            // featNum = sift->GetFeatureNum();

            // //allocate memory
            // keys.resize(featNum);    descriptors.resize(128*featNum);

            // //reading back feature vectors is faster than writing files
            // //if you dont need keys or descriptors, just put NULLs here
            // sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
            // //this can be used to write your own sift file.

        }
        delete sift;
    }


    return 1;
}

template<class Derived>
void imretFuncs::get_descs_for_vv(int& N,  Eigen::MatrixBase<Derived>& desc2vv) {
    //load all descs in one matrix
    std::vector<std::string> siftFiles = stlplus::folder_wildcard(_env.sim_path, "*.sift", false, true);
    std::sort(siftFiles.begin(), siftFiles.end(), compareNat);
    int growsize = 1000000;
    MatrixXfr tmpdesc(growsize, 128);
    MatrixXfr _desc2vv;
    int i;
    int act;
    act = 0;
    for (i = 0 ; i < siftFiles.size(); ++i){
        std::string fullPathSift = _env.sim_path + siftFiles[i];
        //load descs
        MatrixXfr desc_perImg;
        load_descs(fullPathSift, desc_perImg);
        // std::ofstream out_sift(siftFiles[i]+"_readable.txt");
        // if(out_sift.is_open()){
        //         out_sift << desc_perImg << std::endl;
        // }
        // out_sift.close();
        if(act+desc_perImg.rows()>growsize){
            tmpdesc.resize( desc_perImg.rows()+growsize,Eigen::NoChange);
        }



        tmpdesc.block(act,0, desc_perImg.rows(),128) = desc_perImg;

        act += desc_perImg.rows();
        // char * name = abi::__cxa_demangle(typeid(descsReinterpret).name() , 0, 0, 0);
        // std::cout << name << std::endl;
        // free(name);

    }

    _desc2vv = tmpdesc.block(0,0, act,128).eval();
    // int zeros = 0;
    // for(int i = 0; i < _desc2vv.rows(); ++i){
    //         for(int j = 0; j < _desc2vv.cols(); ++j){

    //                 if(_desc2vv(i,j) == 0){

    //                         zeros++;
    //                 }
    //         }

    // }

    //        _desc2vv = _desc2vv.transpose().eval();

    // std::cout << "zeros" << zeros << std::endl;
    std::ofstream desc2vv_file("desc2vv", std::ios::out | std::ios::binary );
    //Write the number of descriptor
    int rows = _desc2vv.rows(), cols = _desc2vv.cols();
    desc2vv_file.write((char*) (&rows), sizeof(int));
    desc2vv_file.write((char*) (&cols), sizeof(int));
    desc2vv_file.write((char*) _desc2vv.data(), rows*cols*sizeof(float) );
    desc2vv_file.close();

    // std::ofstream desc2vv_readable("desc2vv.txt", std::ios::out );
    // desc2vv_readable << rows << cols << _desc2vv;
    // desc2vv_readable.close();

    //Read descriptors
    // std::ifstream in(filename,ios::in | std::ios::binary);
    // in.read((char*) (&rows),sizeof( MatrixXfr::Index));
    // in.read((char*) (&cols),sizeof(MatrixXfr::Index));
    // matrix.resize(rows, cols);
    // in.read( (char *) matrix.data() , rows*cols*sizeof(MatrixXfr::Scalar) );
    // in.close();

    N = siftFiles.size();
    Eigen::MatrixBase<Derived>& desc2vv_ = const_cast<Eigen::MatrixBase<Derived>& >(desc2vv);
    desc2vv_ =  _desc2vv;
}

void imretFuncs::create_vv( const imretDataTypes::opt& opt){
    MatrixXfr desc2vv;
    int N;
    get_descs_for_vv(N, desc2vv);
    int k = std::min(std::floor( desc2vv.rows()*opt.vvsizeratio), (double)opt.vvmaxsize);
    int idxM = 0;
    std::vector<imretDataTypes::vv_struct> tmp(opt.vvrepeats,  imretDataTypes::vv_struct());
    int cnt;

    for (cnt = 0; cnt<opt.vvrepeats; ++cnt){
        flann_approx_kmeans(desc2vv, tmp[cnt].CX, k, tmp[cnt].sses, tmp[cnt].CN, tmp[cnt].assgn);
        if(tmp[cnt].sses.back() <= tmp[idxM].sses.back()){
            idxM = cnt;
        }
        std::cout << "cnt "<<cnt <<" idxM: " << idxM<< std::endl;
    }

    MatrixXfr CX  = tmp[idxM].CX;
    std::vector<float> sses = tmp[idxM].sses;
    std::vector<int> CN = tmp[idxM].CN;
    std::vector<int> assgn = tmp[idxM].assgn;

    std::ofstream vv_file( opt.sadr_vv, std::ios::out | std::ios::binary );
    //Write the number of descriptor
    int rows = CX.rows(), cols = CX.cols();
    int CN_size = CN.size(), assgn_size = assgn.size();

    vv_file.write((char*) (&rows), sizeof(int));
    vv_file.write((char*) (&cols), sizeof(int));
    // vv_file.write((char*) sses.size(), sses.size()*sizeof(int));
    vv_file.write((char*) (&CN_size), sizeof(int));
    vv_file.write((char*) (&N), sizeof(int));
    vv_file.write((char*) (&assgn_size), sizeof(int));


    vv_file.write((char*) CX.data(), CX.rows()*CX.cols()*sizeof(float) );
    vv_file.write((char*) sses.data(),sses.size()*sizeof(float) );
    vv_file.write((char*) CN.data(), CN.size()*sizeof(int));
    vv_file.write((char*) assgn.data(), assgn.size()*sizeof(int));
    vv_file.close();
}



void imretFuncs::simquant( std::vector<std::string>& img_list, const imretDataTypes::opt& opt, std::vector<int>& assgn, std::vector<int>& vw2im){


    std::string filename = opt.sadr_vv;
    int  rows, cols;
    int CN_size;
    int N;
    //Read visaul vocabulary
    std::ifstream in(filename,ios::in | std::ios::binary);
    in.read((char*) (&rows),sizeof(int));
    in.read((char*) (&cols),sizeof(int));
    in.read((char*) (&CN_size),sizeof(int));
    in.read((char*) (&N),sizeof(int));


    MatrixXfr  _CX(rows, cols);
    std::vector<int> CN(CN_size);
    in.read( (char *) _CX.data() , rows*cols*sizeof(float) );
    in.read( (char *) CN.data() , CN_size*sizeof(int) );
    in.close();


    // flann parameters
    int checks  = 232, trees = 1,branching = 32,iterations= 5;

    flann::CompositeIndexParams kmeans_param(trees, branching,iterations, flann::FLANN_CENTERS_RANDOM, 0.2);


    //flann search

    flann::Matrix<Scalar> CX((Scalar*)  _CX.data(), rows, cols);

    imretDataTypes::vv_flann vv;
    vv.fl_params = checks;


    std::unique_ptr< flann::Matrix<int> > indices;
    indices.reset(new flann::Matrix<int>(new int[CX.rows], CX.rows, 1));
    std::unique_ptr<flann::Matrix<DistanceType> > dists;
    dists.reset(new flann::Matrix<DistanceType>(new float[CX.rows], CX.rows, 1));

    //-- Build FLANN index
    vv.fl_idx.reset(new flann::Index<Metric> (CX, kmeans_param));
    (vv.fl_idx)->buildIndex();


    std::vector<std::vector<int> > assgnimg;
    std::vector<std::vector<int> > vw2imimgs;

    int i;

    for(i = 0; i <img_list.size(); ++i){
        std::string dvw_fname =  _env.sim_path + stlplus::basename_part(img_list[i]);
        std::vector<int> assgnimg_perimg;
        sift2dvw(dvw_fname,vv,assgnimg_perimg);
        // int maxAssgn =  *std::max_element(assgnimg_perimg.begin(), assgnimg_perimg.end());
        // std::cout << i << " " << maxAssgn << std::endl;
        assgnimg.push_back(assgnimg_perimg);
    }
    vv.fl_idx.reset();

    std::vector<int> assgn_, vw2im_;

    for(i = 0; i< assgnimg.size(); ++i){
        std::vector<int> vw2imimg(assgnimg[i].size(), i);
        std::copy(assgnimg[i].begin(), assgnimg[i].end(),std::back_inserter(assgn_));
        std::copy(vw2imimg.begin(), vw2imimg.end(),std::back_inserter(vw2im_));
    }


    // std::copy(assgn_.begin(), assgn_.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    // std::cout << "mark ftrs_to_sim" << std::endl;

    std::string ftrs_to_sim_name = _env.sim_path + "ftrs_to_sim";
    std::ofstream ftrs_to_sim(ftrs_to_sim_name, std::ios::out | std::ios::binary);
    int assgn_size = assgn_.size(), vw2im_size = vw2im_.size();


    ftrs_to_sim.write((char*) (&assgn_size), sizeof(int));
    ftrs_to_sim.write((char*) (&vw2im_size), sizeof( int));
    ftrs_to_sim.write((char*) (&CN_size), sizeof( int));
    ftrs_to_sim.write((char*) (&N), sizeof(int));

    ftrs_to_sim.write((char*) assgn_.data(), assgn_size*sizeof(int) );
    ftrs_to_sim.write((char*) vw2im_.data(), vw2im_size*sizeof(int) );
    ftrs_to_sim.write((char*) CN.data(), CN_size*sizeof(int) );


    ftrs_to_sim.close();
    assgn = assgn_;
    vw2im = vw2im_;
}

void imretFuncs::sift2dvw(const std::string& dvw_fname, const imretDataTypes::vv_flann& vv, std::vector<int>& assgnimg_perimg){

    std::string dvw_fname_sift = dvw_fname + ".sift";
    std::string dvw_fname_assgn = dvw_fname + ".assgn";


    if(stlplus::file_exists(dvw_fname_assgn)){
        int assgn_size;
        std::ifstream in(dvw_fname_assgn,ios::in | std::ios::binary);
        in.read((char*) (&assgn_size),sizeof(int));
        std::vector<int> assgnimg_perimg_copy(assgn_size);
        in.read( (char *) assgnimg_perimg_copy.data() , assgn_size*sizeof(int) );
        in.close();
        assgnimg_perimg = assgnimg_perimg_copy;

    }else{
        MatrixXfr descs_mat;
        load_descs(dvw_fname_sift, descs_mat);
        flann::Matrix<Scalar> descs_mat_flann;
        convert2flann(descs_mat, descs_mat_flann);

        std::unique_ptr< flann::Matrix<int> > indices;
        indices.reset(new flann::Matrix<int>(new int[descs_mat_flann.rows], descs_mat_flann.rows, 1));
        std::unique_ptr<flann::Matrix<DistanceType> > dists;
        dists.reset(new flann::Matrix<DistanceType>(new float[descs_mat_flann.rows], descs_mat_flann.rows, 1));
        flann::SearchParams simquant_search_params;
        simquant_search_params.checks =vv.fl_params;
        simquant_search_params.cores = 0;
        //        (vv.fl_idx)->knnSearch(descs_mat_flann, *indices, *dists, 1, flann::SearchParams(vv.fl_params));
        (vv.fl_idx)->knnSearch(descs_mat_flann, *indices, *dists, 1, simquant_search_params);
        std::vector<int> indices_vec(indices->ptr(), indices->ptr()+(indices->cols));


        //write to file
        std::ofstream dvw_file(dvw_fname_assgn, std::ios::out | std::ios::binary );
        int indices_vec_size = indices_vec.size();
        dvw_file.write((char*) (&indices_vec_size), sizeof(int));
        dvw_file.write((char*) indices_vec.data(), indices_vec_size*sizeof(int) );
        dvw_file.close();
        assgnimg_perimg = indices_vec;
    }
}

/*
template<class Derived>
void imretFuncs::simget(const Eigen::SparseMatrixBase<Derived>& sim ){
        std::string imsim_name = _env.sim_path+ "imsim";
        bool imsim_exist = stlplus::file_exists(imsim_name);
        const Derived &sim_(sim.derived());
        Eigen::SparseMatrix<typename Derived::Scalar> _sim = sim_;
        if(imsim_exist){
                //need to load sim
                std::ifstream in_file( imsim_name, std::ios::in | std::ios::binary);
                int in_val_size, in_inner_size,in_outer_size, in_rows, in_cols;

                in_file.read((char*) (&in_val_size), sizeof(int));
                in_file.read((char*) (&in_inner_size), sizeof(int));
                in_file.read((char*) (&in_outer_size), sizeof(int));
                in_file.read((char*) (&in_rows), sizeof(int));
                in_file.read((char*) (&in_cols), sizeof(int));

                std::vector<float> values(in_val_size);
                std::vector<int> inner_indices(in_inner_size), outer_indices(in_outer_size);
                in_file.read((char*) values.data(), in_val_size*sizeof(float));
                in_file.read((char*) inner_indices.data(), in_inner_size*sizeof(int));
                in_file.read((char*) outer_indices.data(), in_outer_size*sizeof(int));
                in_file.close();

                std::vector<Tri> tripletList;
                tripletList.reserve(in_val_size);
                int i,j;
                for(i = 0; i < in_inner_size; ++i){
                        for(j = 0; j < in_outer_size; ++j){
                                tripletList.push_back(Tri(inner_indices[i],j,values[i*in_inner_size+j]));
                        }
                }

                _sim =  Eigen::SparseMatrix<float>(in_rows, in_cols);
                _sim.setFromTriplets(tripletList.begin(), tripletList.end());

        }else{
                //load ftrs_to_sim

                std::string ftrs_to_sim_name = _env.sim_path + "ftrs_to_sim";
                std::ifstream in(ftrs_to_sim_name, std::ios::in | std::ios::binary);
                int assgn_size, vw2im_size, CN_size;
                int N;

                in.read((char*) (&assgn_size),sizeof(int));
                in.read((char*) (&vw2im_size),sizeof(int));
                in.read((char*) (&CN_size),sizeof(int));
                in.read((char*) (&N),sizeof(int));

                std::vector<int> assgn(assgn_size), vw2im(vw2im_size),  CN(CN_size);
                in.read( (char *) assgn.data() , assgn_size*sizeof(int) );
                in.read( (char *) vw2im.data() , vw2im_size*sizeof(int) );
                in.read( (char *) CN.data() , CN_size*sizeof(int) );
                in.close();

                imretDataTypes::data_qr qr_tfidf;
                imretDataTypes::data_vv vv_tfidf;
                qr_tfidf.assgn = assgn;
                qr_tfidf.vw2im = vw2im;
                vv_tfidf.CN = CN;
                vv_tfidf.N = N;
                Eigen::SparseMatrix<float> tfidf;
                tfidf_from_vvQR_fast(qr_tfidf, vv_tfidf, tfidf);

                _sim = tfidf.transpose()*tfidf;


                // char * name = abi::__cxa_demangle(typeid(sim.nonZeros()).name() , 0, 0, 0);
                // std::cout << name << std::endl;
                // free(name);

                //need to save sim
                std::ofstream sim_file( imsim_name, std::ios::out | std::ios::binary);
                int sim_val_size = _sim.nonZeros(), inner_size = _sim.innerSize(), outer_size = _sim.outerSize(), rows = _sim.rows(), cols = _sim.cols();

                sim_file.write((char*) (&sim_val_size), sizeof(int));
                sim_file.write((char*) (&inner_size), sizeof(int));
                sim_file.write((char*) (&outer_size), sizeof(int));
                sim_file.write((char*) (&rows), sizeof(int));
                sim_file.write((char*) (&cols), sizeof(int));

                sim_file.write((char*) _sim.valuePtr(), sim_val_size*sizeof(int));
                sim_file.write((char*) _sim.innerIndexPtr(), inner_size*sizeof(int));
                sim_file.write((char*) _sim.outerIndexPtr(), outer_size*sizeof(int));
                sim_file.close();

                // for(int i = 0; i< in_val_size; ++i){
                //         std::cout<< outer_indices[i]<< " "  << inner_indices[i] << " " << values[i] << std::endl;
                // }

        }

        // Eigen::SparseMatrixBase<Derived>& sim_ = const_cast< Eigen::SparseMatrixBase<Derived>& >(sim);
        // sim_ = _sim;

}
*/

template<class Derived>
void imretFuncs::tfidf_from_vvQR_fast(const imretDataTypes::data_qr& qr_tfidf, const imretDataTypes::data_vv& vv_tfidf, Eigen::SparseMatrixBase<Derived>& tfidf ){

    std::vector<Tri> tripletList;

    int maxAssgn =  *std::max_element(qr_tfidf.assgn.begin(), qr_tfidf.assgn.end());
    int maxVw2im = *std::max_element(qr_tfidf.vw2im.begin(), qr_tfidf.vw2im.end());

    // std::cout << maxAssgn << std::endl;
    // std::copy(qr_tfidf.assgn.begin(), qr_tfidf.assgn.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    // int sth = 0, idx_sth = 0;
    // for(int i = 0; i < qr_tfidf.assgn.size(); ++i){
    //         if(qr_tfidf.assgn[i] > sth){
    //                 sth = qr_tfidf.assgn[i];
    //                 idx_sth = i;
    //         }
    // }
    // std::cout << qr_tfidf.assgn.size() << " " <<sth << " " << idx_sth << std::endl;
    tripletList.reserve(qr_tfidf.assgn.size()+100);
    //std::cout << " it works here 1" << std::endl;


    int i,j;
    for(i = 0; i < qr_tfidf.assgn.size(); ++i){
        tripletList.push_back(Tri(qr_tfidf.assgn[i],qr_tfidf.vw2im[i],1));
    }



    int rowDiff = vv_tfidf.CN.size() - maxAssgn-1;

    if(rowDiff > 0){
        for(i = 1; i <= rowDiff; ++i){
            tripletList.push_back(Tri(maxAssgn+i, maxVw2im, 0));
        }
    }



    Eigen::SparseMatrix<float> tf(vv_tfidf.CN.size(), maxVw2im+1);
    tf.setFromTriplets(tripletList.begin(), tripletList.end());



    std::vector<float> nd;
    for(i = 0; i < tf.cols(); ++i){
        float my_sum = 0;
        float *val_ptr = tf.col(i).valuePtr();

        if(tf.col(i).nonZeros()){
            // avoid division by zero problem
            for (j = 0; j < tf.col(i).nonZeros(); ++j){
                my_sum += *val_ptr;
                val_ptr++;
            }
            nd.push_back(1/my_sum);
        }else{
            nd.push_back(0);
        }
    }



    std::vector<float> idf(vv_tfidf.CN.size());

    //problem with CN

    for(i = 0; i < vv_tfidf.CN.size();++i){
        //                assert(vv_tfidf.CN[i] == 0);
        if(vv_tfidf.CN[i]!=0){
            idf[i] =log( (float)vv_tfidf.N/(float)vv_tfidf.CN[i]);
            // }else{

            //         idf[i] =log( (float)vv_tfidf.N);
        }
    }



    std::vector<Tri> tripletList_tmp1, tripletList_tmp2;
    tripletList_tmp1.reserve(idf.size());
    tripletList_tmp2.reserve(nd.size());

    for(i = 0; i < idf.size(); ++i){
        tripletList_tmp1.push_back(Tri(i,i,idf[i]));

    }
    for(i = 0; i < nd.size(); ++i) {
        tripletList_tmp2.push_back(Tri(i,i,nd[i]));
    }


    Eigen::SparseMatrix<float> tmp1(idf.size(),idf.size()), tmp2(nd.size(), nd.size());
    tmp1.setFromTriplets(tripletList_tmp1.begin(), tripletList_tmp1.end());
    tmp2.setFromTriplets(tripletList_tmp2.begin(), tripletList_tmp2.end());
    Eigen::SparseMatrix<float> _tfidf = tmp1*tf*tmp2;

    //normalize
    std::vector<float> nrm;
    for (i = 0; i < _tfidf.cols(); ++i){
        float my_sum = 0;
        float *val_ptr = _tfidf.col(i).valuePtr();

        if(_tfidf.col(i).nonZeros()){
            // avoid division by zero problem

            for (j = 0; j < _tfidf.col(i).nonZeros(); ++j){
                my_sum += (*val_ptr * (*val_ptr));
                val_ptr++;
            }
            nrm.push_back(1.0 / sqrt(my_sum));
        }else{
            nrm.push_back(0);
        }
    }



    std::vector<Tri> tripletList_nrm;
    tripletList_nrm.reserve(_tfidf.cols());
    for(i = 0; i < nrm.size(); ++i){
        tripletList_nrm.push_back(Tri(i,i,nrm[i]));
    }
    Eigen::SparseMatrix<float> sparse_nrm(_tfidf.cols(),_tfidf.cols());
    sparse_nrm.setFromTriplets(tripletList_nrm.begin(), tripletList_nrm.end());
    _tfidf = _tfidf*sparse_nrm;

    Eigen::SparseMatrixBase<Derived>& tfidf_ = const_cast< Eigen::SparseMatrixBase<Derived>& >(tfidf);
    tfidf_ = _tfidf;

}



template<class Derived>
void  imretFuncs::flann_approx_kmeans(const Eigen::MatrixBase<Derived>& X,   Eigen::MatrixBase<Derived>& CX,  int nclus,  std::vector<float>& sses, std::vector<int>& CN,  std::vector<int>& assgn){

    std::cout << "k "<< nclus << std::endl;
    // X is 128xCols
    //permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.rows());
    perm.setIdentity();

    std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());

    MatrixXfr _tmpCX = perm*X;



    MatrixXfr _tmpCX_rm = _tmpCX.block(0,0,nclus,128).eval();
    flann::Matrix<Scalar> _CX((Scalar*)  _tmpCX_rm.data(),  nclus,128);
    MatrixXfr _tmpX = X;
    flann::Matrix<Scalar>_X((Scalar*)_tmpX.data(), _tmpX.rows(),128);

    int maxiters = 20;
    int iter = 0;
    //float mindelta = std::numeric_limits<float>::epsilon();
    float mindelta = 0.01;
    //        float mindelta = 5;
    float sse0 = std::numeric_limits<int>::max();

    while (iter < maxiters){
        std::cout << "iter " << iter << std::endl;
        std::unique_ptr< flann::Index<Metric> > _index;
        std::unique_ptr< flann::Matrix<int> > indices;
        indices.reset(new flann::Matrix<int>(new int[_X.rows], _X.rows, 1));
        std::unique_ptr<flann::Matrix<DistanceType> > dists;
        dists.reset(new flann::Matrix<DistanceType>(new float[_X.rows], _X.rows, 1));

        //-- Build FLANN index
        _index.reset(new flann::Index<Metric> (_CX, flann::KDTreeIndexParams(5)));
        _index->buildIndex();


        //do a knn search, using 128 checks

        flann::SearchParams kmeans_search_params;
        kmeans_search_params.checks = 128;
        kmeans_search_params.cores = 0;

        _index->knnSearch(_X, *indices, *dists, 1, kmeans_search_params);
        _index.reset();


        std::vector<int> indices_vec(indices->ptr(), indices->ptr()+indices->rows);

        std::vector<size_t> indices_ind;
        std::vector<int> indices_sorted;
        this->sort(indices_vec,indices_sorted,indices_ind);

        // std::copy(indices_vec.begin(), indices_vec.end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;
        std::vector<int> sidx(indices_vec.size()+1), eidx(indices_vec.size()+1);
        sidx[0] = indices_sorted[0]+1;
        eidx[0] = indices_sorted[0];
        std::copy(indices_sorted.begin(), indices_sorted.end()-1, sidx.begin()+1);
        std::transform(indices_sorted.begin()+1, indices_sorted.end(), indices_sorted.begin(), sidx.begin()+1, [](float a, float b) { return (a-b); });
        sidx.back() = nclus - indices_sorted.back()-1;
        //                std::copy(sidx.begin(), sidx.end(), std::ostream_iterator<int>(std::cout, " "));
        std::copy(sidx.begin()+1, sidx.end()-1,eidx.begin()+1 );
        eidx.back() = nclus-indices_sorted.back();

        int maxSidx =  *std::max_element(sidx.begin(), sidx.end());
        int maxEidx = *std::max_element(eidx.begin(), eidx.end());
        std::vector<int> sidxf, eidxf;

        int i,j;
        for(i = 1; i <= maxSidx; ++i){
            for(j = 0; j < sidx.size();++j){
                if(sidx[j]>=i){
                    sidxf.push_back(j);
                }

            }
            std::sort(sidxf.begin(), sidxf.end());
        }



        for(i = 1; i <= maxEidx; ++i){
            for(j = 0; j < sidx.size();++j){
                if(eidx[j] >= i){
                    eidxf.push_back(j-1);
                }

            }

            std::sort(eidxf.begin(), eidxf.end());
        }


        // std::copy(sidxf.begin(), sidxf.end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;
        // std::copy(eidxf.begin(), eidxf.end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;

        MatrixXfr _tmpCX_float = MatrixXfr::Zero(_tmpCX_rm.rows(), _tmpCX_rm.cols());

        for (i = 0; i < sidxf.size(); ++i){
            for(j = sidxf[i]; j < eidxf[i]; ++j ) {
                _tmpCX_float.row(i) +=  _tmpX.row(indices_ind[j]);
            }
        }

        _tmpCX_float/=(float)sidxf.size();


        int maxIdxFlann =  *std::max_element(indices_sorted.begin(), indices_sorted.end());

        std::vector<int> CN_temp(maxIdxFlann+1, 0);
        for(i = 0; i< indices_vec.size(); ++i){
            CN_temp[indices_vec[i]]++;
        }

        MatrixXfr _tmpX_float = _tmpX;
        for(i =0; i< indices_vec.size(); ++i){
            _tmpX_float.row(i)-=_tmpCX_float.row(indices_vec[i]);
        }

        _tmpX_float =  _tmpX_float.array().square();

        MatrixXfr _tmpX_float_sum = _tmpX_float.rowwise().sum();
        float sse =  _tmpX_float_sum.array().sqrt().sum();
        // std::cout << X.cols() << std::endl;
        //std::cout << sse << std::endl;
        sse /=(float)X.rows();
        //std::cout <<sse << std::endl;
        //                _tmpX_float = _tmpX;


        std::vector<int> empts;

        for (i = 0; i < CN_temp.size(); ++i){
            if(CN_temp[i] == 0){
                empts.push_back(i);
            }
        }

        // std::copy(empts.begin(), empts.end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;
        if(empts.size() > 0){
            std::cout <<"there were " << empts.size() << " non-assigned visual words"<<std::endl;

            for(i = 0; i < empts.size(); ++ i ){
                std::vector<int> uses;
                for(j = 0; j < CN_temp.size(); ++j){
                    if(CN_temp[j] > 1){
                        uses.push_back(j);
                    }
                }

                int idpt = std::floor(static_cast <float>(rand())/static_cast<float>(RAND_MAX)*static_cast<float>(indices_vec.size()-1));
                int assgnEqUses = 0;

                for(j = 0; j < uses.size(); ++j ){
                    if(indices_vec[idpt] == uses[j]){
                        assgnEqUses++;
                        break;
                    }
                }

                while(assgnEqUses == 0){
                    idpt = std::floor(static_cast <float>(rand())/static_cast<float>(RAND_MAX)*static_cast<float>(indices_vec.size()-1));
                    for(j = 0; j < uses.size(); ++j ){
                        if(indices_vec[idpt] == uses[j]){
                            assgnEqUses++;
                            break;
                        }
                    }

                }

                CN_temp[indices_vec[idpt]] = CN_temp[indices_vec[idpt]] -1;
                CN_temp[empts[i]]  = 1;
                indices_vec[idpt] = empts[i];
                _tmpCX_float.row(empts[i])=_tmpX_float.row(idpt);

            }
        }

        sses.push_back(sse);

        //                std::cout << std::abs( sse0 - sse ) << " " << mindelta << std::endl;
        //                if((sse0 - sse < mindelta)&& (sse0 - sse > 0)){
        if(std::abs(sse0 - sse) < mindelta){
            CN = CN_temp;
            assgn = indices_sorted;
            Eigen::MatrixBase<Derived>& CX_ = const_cast< Eigen::MatrixBase<Derived>& >(CX);
            CX_ =  _tmpCX_float;
            break;
        }

        sse0=sse;
        iter++;

        if(iter==maxiters){
            CN = CN_temp;
            assgn = indices_sorted;
            Eigen::MatrixBase<Derived>& CX_ = const_cast< Eigen::MatrixBase<Derived>& >(CX);
            CX_ =  _tmpCX_float;
        }


        //                std::copy(CN_temp.begin(), CN_temp.end(), std::ostream_iterator<int>(std::cout, " "));
        //std::cout<<_tmp_mat.array().sqrt()<<std::endl;
        // std::copy(indices_sorted.begin(), indices_sorted.end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;

    }
}




template <class T>
void imretFuncs::sort(std::vector<T> & unsorted,std::vector<T> & sorted,std::vector<size_t> & index_map){
    // Original unsorted index map
    index_map.resize(unsorted.size());
    for(size_t i=0;i<unsorted.size();i++){
        index_map[i] = i;
    }
    // Sort the index map, using unsorted for comparison
    std::sort(index_map.begin(), index_map.end(), imretDataTypes::index_cmp<std::vector<T>& >(unsorted));
    sorted.resize(unsorted.size());
    imretFuncs::reorder(unsorted,index_map,sorted);
}


// This implementation is O(n), but also uses O(n) extra memory
template< class T >
void imretFuncs::reorder(std::vector<T> & unordered, std::vector<size_t> const & index_map, std::vector<T> & ordered){
    // copy for the reorder according to index_map, because unsorted may also be
    // sorted
    std::vector<T> copy = unordered;
    ordered.resize(index_map.size());
    for(size_t i = 0; i<index_map.size();i++){
        ordered[i] = copy[index_map[i]];
    }
}

void imretFuncs::getEnv(imretDataTypes::ENV& env) const{
    env.sim_path = this->_env.sim_path;
    env.img_path = this->_env.img_path;
    env.imfmt = this->_env.imfmt;

}

void imretFuncs::load_ftrs_to_sim( std::vector<int>& assgn, std::vector<int>& vw2im, std::vector<int>& CN, int& N){

    std::string ftrs_to_sim_name = _env.sim_path + "ftrs_to_sim";

    std::ifstream in(ftrs_to_sim_name, std::ios::in | std::ios::binary);
    int assgn_size, vw2im_size, CN_size;

    in.read((char*) (&assgn_size),sizeof(int));
    in.read((char*) (&vw2im_size),sizeof(int));
    in.read((char*) (&CN_size),sizeof(int));
    in.read((char*) (&N),sizeof(int));


    std::vector<int> assgn_(assgn_size), vw2im_(vw2im_size),  CN_(CN_size);
    in.read( (char *) assgn_.data() , assgn_size*sizeof(int) );
    in.read( (char *) vw2im_.data() , vw2im_size*sizeof(int) );
    in.read( (char *) CN_.data() , CN_size*sizeof(int) );
    in.close();

    // std::copy(assgn_.begin(), assgn_.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    assgn=assgn_;
    vw2im = vw2im_;
    CN = CN_;

}

template<class Derived>
void imretFuncs::convert2flann(const Eigen::MatrixBase<Derived>& mat_eigen,flann::Matrix<Scalar>& mat_flann){
    MatrixXfr mat_eigen_float(mat_eigen);
    mat_flann = flann::Matrix<Scalar> ((Scalar*) mat_eigen_float.data(), mat_eigen_float.rows(), mat_eigen_float.cols());
}
template<class Derived>
void imretFuncs::load_descs(const std::string& dvw_fname_sift, Eigen::MatrixBase<Derived>& descs_mat){
    //Read descriptors
    std::string fileName =  dvw_fname_sift;
    //        std::cout << fileName << std::endl;
    int rows , cols ;
    std::ifstream in(fileName,ios::in | std::ios::binary);
    in.read((char*) (&rows),sizeof(int));
    in.read((char*) (&cols),sizeof(int));
    //        std::cout << rows <<" "<<cols<<std::endl;
    MatrixXfr _descs_mat(rows,cols);
    for(int i = 0; i < rows; ++ i){
        float * feature = new float[4];
        in.read((char*) (feature), 4*sizeof(float));
        delete feature;
        MatrixXfr  descs_row(1, cols);
        in.read( (char *) (descs_row.data()) , cols*sizeof(float));
        _descs_mat.row(i) =  descs_row.row(0).eval();
    }
    in.close();
    Eigen::MatrixBase<Derived>& descs_mat_ = const_cast<Eigen::MatrixBase<Derived>& >(descs_mat);
    descs_mat_ = _descs_mat;


}

template<class Derived>
void imretFuncs::load_feats(const std::string& dvw_fname_sift,  Eigen::MatrixBase<Derived> & feats_mat){

    //Read descriptors
    std::string fileName =  dvw_fname_sift;
    int rows , cols ;
    std::ifstream in(fileName,ios::in | std::ios::binary);
    in.read((char*) (&rows),sizeof(int));
    in.read((char*) (&cols),sizeof(int));
    MatrixXfr _feats_mat(rows,4);
    for(int i = 0; i < rows; ++ i){
        MatrixXfr feature(1,4);
        in.read((char*) (feature.data()), 4*sizeof(float));
        _feats_mat.row(i) = feature.row(0).eval();
        MatrixXfr  descs_row(1, cols);
        in.read( (char *) (descs_row.data()) , cols*sizeof(float));
    }
    in.close();
    Eigen::MatrixBase<Derived>& feats_mat_ = const_cast<Eigen::MatrixBase<Derived>& >(feats_mat);
    feats_mat_ = _feats_mat;

}

void imretFuncs::get_sim_path(std::string& sim_path){
    sim_path = _env.sim_path;
}

imretFuncs::~imretFuncs(){
    FREE_MYLIB(hsiftgpu);
}

//////////////////////////////////////////////////////////////////////////standalone functions/////////////////////////////////////////////////////////////////////////
bool compareNat(const std::string& a, const std::string& b){
    if (a.empty())
        return true;
    if (b.empty())
        return false;
    if (std::isdigit(a[0]) && !std::isdigit(b[0]))
        return true;
    if (!std::isdigit(a[0]) && std::isdigit(b[0]))
        return false;
    if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
    {
        if (a[0] == b[0])
            return compareNat(a.substr(1), b.substr(1));
        return (std::toupper(a[0]) < std::toupper(b[0]));
    }

    // Both strings begin with digit --> parse both numbers
    std::istringstream issa(a);
    std::istringstream issb(b);
    int ia, ib;
    issa >> ia;
    issb >> ib;
    if (ia != ib)
        return ia < ib;

    // Numbers are the same --> remove numbers and recurse
    std::string anew, bnew;
    std::getline(issa, anew);
    std::getline(issb, bnew);
    return (compareNat(anew, bnew));
}

bool is_not_digit(char c){

    return !std::isdigit(c);
}

bool numeric_string_compare(const std::string& s1, const std::string& s2){
    std::string::const_iterator it1 = s1.begin(), it2 = s2.begin();
    std::vector<std::string> lineElems1, lineElems2;
    // boost::split(lineElems1, s1, boost::is_any_of("."));
    // boost::split(lineElems2, s2, boost::is_any_of("."));

    // lineElems
    if(std::isdigit(s1[0] && std::isdigit(s2[0]))){
        int n1, n2;
        std::stringstream ss(s1);
        ss >> n1;
        ss.clear();
        ss.str(s2);
        ss >> n2;
        if(n1 != n2){
            return n1 < n2;
        }

        it1 = std::find_if(s1.begin(), s1.end(), is_not_digit);
        it2 = std::find_if(s2.begin(), s2.end(), is_not_digit);

    }

    return std::lexicographical_compare(it1, s1.end(), it2, s2.end());
}


void get_labels_db(const std::string& labelPath, std::vector<std::vector<double> >& labels){
    std::ifstream inLabelFile(labelPath);
    std::vector<std::vector<double>> labels_;
    std::string line;
    int i;
    while(std::getline(inLabelFile, line)){
        if(line !="" && line != "\n" && line != "\t"&& line!= " "){
            std::istringstream iss(line);
            std::vector<double> lineElems(6,0);
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            labels_.push_back(lineElems);
        }
    }
    inLabelFile.close();

    labels = labels_;
}


void getDBImID(const std::string& gps_proc_path, const std::string& label_db_path, std::vector<std::vector<double> >& gps_assgn){
    //void getDBImID(const std::string& gps_db_path, const std::string& gps_qr_path, const std::string& label_db_path, std::vector<std::vector<double> >& gps_assgn){

    //        getGPSsimil( gps_db_path,  gps_qr_path);
    std::ifstream inGPSFile(gps_proc_path);
    std::ifstream inLabelFile(label_db_path);
    std::vector<std::vector<double> > gps_assgn_;
    std::string line;
    int i;


    while(std::getline(inLabelFile, line)){
        if(line !="" && line != "\n" && line != "\t"&& line!= " "){
            std::istringstream iss(line);
            std::vector<double> lineElems(6,0);
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            gps_assgn_.push_back(lineElems);
        }
    }
    inLabelFile.close();

    // std::string delimiter = "\r";
    // std::ofstream outLabelFile("labels.txt");
    // while(std::getline(inLabelFile, line, '\r')){

    //         std::string token = line.substr(0, line.find(delimiter));
    //         std::string newline = token + '\n';
    //         outLabelFile << newline;

    // }
    // outLabelFile.close();

    while(std::getline(inGPSFile, line)){
        if(line !="" && line != "\n" && line != "\t"&& line!= " "){
            std::istringstream iss(line);
            std::vector<double> lineElems(4,0);
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            for(i = 0; i < gps_assgn_.size(); ++i ){
                if(gps_assgn_[i].size() < 7){
                    //                         if(gps_assgn_[i].size() < 9){
                    //                                 if(gps_assgn_[i][0] == lineElems.back() || (gps_assgn_[i][0] == lineElems.back() + 1) || (gps_assgn_[i][0] == lineElems.back() - 1)){
                    if(gps_assgn_[i][0] == lineElems.back()){
                        gps_assgn_[i].push_back(lineElems.front());
                    }
                }
            }
        }
    }
    inGPSFile.close();

    for(i = 0; i < gps_assgn_.size(); ++i){
        if(gps_assgn_[i].size() < 7){
            //                   if(gps_assgn_[i].size() < 9){
            gps_assgn_[i].push_back(0);
        }
    }


    //being revised///
    std::string workspace =stlplus::folder_part(gps_proc_path);
    std::string outFilePath = workspace + "/"+stlplus::basename_part(gps_proc_path)+"_assgn.txt";
    std::ofstream out_gps(outFilePath,std::ofstream::out);

    for(i = 0; i < gps_assgn_.size(); ++i){
        std::copy( gps_assgn_[i].begin(), gps_assgn_[i].end(), std::ostream_iterator<double>(out_gps, "\t"));
        out_gps << std::endl;
    }

    gps_assgn = gps_assgn_;
}


void getGazes(const std::string& gazes_path, std::vector<std::vector<double> >& gazes){
    std::ifstream inGazesFile(gazes_path, std::ifstream::in);
    std::string line;
    std::vector<std::vector<double> > gazes_;
    while(std::getline(inGazesFile, line)){
        if(line!=""){
            std::istringstream iss(line);
            std::vector<double> lineElems(4,0);
            int i;
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            gazes_.push_back(lineElems);
        }

    }
    inGazesFile.close();
    gazes = gazes_;

}
void get_gps_qr(const std::string qr_label_path,const std::string gps_qr_path, std::vector<std::vector<double> >& labels ){
    std::ifstream inGPSFile(gps_qr_path, std::ifstream::in);
    std::ifstream inLabelPath(qr_label_path, std::ifstream::in);

    std::string line;
    std::vector<std::vector<double>> GPS_;
    std::vector<std::vector<double>> labels_;
    while(std::getline(inGPSFile, line)){
        if(line !=""){
            std::istringstream iss(line);
            std::vector<double> lineElems(3,0);
            int i;
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            GPS_.push_back(lineElems);
        }
    }

    inGPSFile.close();

    while(std::getline(inLabelPath, line)){
        if(line!=""){
            std::istringstream iss(line);
            std::vector<double> lineElems(2,0);
            int i,j;

            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];

            }
            for(i = 0; i < GPS_.size(); ++i){
                if(lineElems[1] == GPS_[i][0] ){
                    for(j = 1; j < GPS_[i].size(); ++j){
                        lineElems.push_back(GPS_[i][j]);
                    }
                }
            }
            labels_.push_back(lineElems);
        }
    }

    inLabelPath.close();
    labels = labels_;
}

double  getDisPnt2Vert(std::pair<double, double> pnt, std::pair<double, double> vert1, std::pair<double, double> vert2){

    std::pair<double, double> v1((pnt.first - vert1.first)*1000000, (pnt.second - vert1.second)*1000000 );
    std::pair<double, double> v2((vert1.first - vert2.first)*1000000, (vert1.second - vert2.second)*1000000);
    double costheta = (v2.first * v1.first + v2.second*v1.second);

    if (costheta < 0){
        return -1;
    }else{
        std::pair<double, double> result;
        result.first = v1.first* (v2.first*v2.first + v2.second*v2.second) - (v1.first*v2.first + v1.second*v2.second)*v2.first;
        result.second = v1.second*(v2.first*v2.first + v2.second*v2.second) - (v1.first*v2.first + v1.second*v2.second)*v2.second;
        return result.first*result.first + result.second*result.second;
    }
}

void getGPSsimil(const std::string& gps_db_path, const std::string& gps_qr_path){

    //check the distance sum from the user location to each pair of endpoints of the bounding box edges of each building
    //find out the smallest one for each query image

    std::ifstream inGPS_qr_file(gps_qr_path, std::ifstream::in);
    std::ifstream inGPS_db_file(gps_db_path, std::ifstream::in);

    std::string line;
    std::vector<std::vector<double>> GPS_qr;
    std::vector<std::vector<double>> GPS_db;
    while(std::getline(inGPS_qr_file, line)){
        if(line !="" && line != "\n" && line != "\t"&& line!= " "){
            std::istringstream iss(line);
            std::vector<double> lineElems(3,0);
            int i;
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            GPS_qr.push_back(lineElems);
        }
    }

    inGPS_qr_file.close();

    while(std::getline(inGPS_db_file, line)){
        if(line !="" && line != "\n" && line != "\t"&& line!= " "){
            std::istringstream iss(line);
            std::vector<double> lineElems(11,0);
            int i;
            for(i = 0; i < lineElems.size(); ++i){
                iss >> lineElems[i];
            }
            GPS_db.push_back(lineElems);
        }
    }

    inGPS_db_file.close();


    int i,j,m,n;

    std::vector<double> dist2centers;
    for(i = 0; i < GPS_qr.size(); ++i){
        std::pair<double, double> qr_one_gps(GPS_qr[i][1], GPS_qr[i][2]);
        double minDist = std::numeric_limits<double>::max();
        double minDist2center = std::numeric_limits<double>::max();
        double featID = 0;
        //                double featID2 = 0;
        for(j = 0; j < GPS_db.size(); ++j){
            double dist2center = (GPS_db[j][9] - qr_one_gps.first)*(GPS_db[j][9]- qr_one_gps.first)+ (GPS_db[j][10]- qr_one_gps.second)*(GPS_db[j][10]-qr_one_gps.second);
            // double current_minDist = minDist;
            // std::vector<std::pair<double, double> > verts;
            // std::vector <double> dist(4,0);
            // for(m = 0; m < 8; m +=2){
            //         std::pair<double, double> db_one_gps(GPS_db[j][1+m], GPS_db[j][2+m]);
            //         verts.push_back(db_one_gps);
            // }

            // dist[0] = getDisPnt2Vert(qr_one_gps, verts[0], verts[1]);
            // dist[1] = getDisPnt2Vert(qr_one_gps, verts[2], verts[1]);
            // dist[2] = getDisPnt2Vert(qr_one_gps, verts[3], verts[2]);
            // dist[3] = getDisPnt2Vert(qr_one_gps, verts[3], verts[4]);

            // for(n = 0 ; n < verts.size(); ++n){
            //         if((current_minDist > dist[n]) && (dist[n] > 0)){
            //                 current_minDist = dist[n];

            //         }
            // }
            // if (( std::abs(minDist - current_minDist) < 0.001) || current_minDist < minDist){
            //         if(dist2center < minDist2center){
            //                 minDist2center = dist2center;
            //                 featID = GPS_db[j][0];
            //                 if(current_minDist < minDist) {
            //                         minDist = current_minDist;                                        }
            //         }

            // }
            if(dist2center < minDist2center){
                featID = GPS_db[j][0];
                minDist2center = dist2center;
            }

        }


        GPS_qr[i].push_back(featID);

    }

    //write 2 file
    std::string workspace =stlplus::folder_part(gps_db_path);
    std::string outFilePath = workspace + "/"+stlplus::basename_part(gps_qr_path)+"_processed.txt";
    std::ofstream out_gps(outFilePath,std::ofstream::out);

    for(i = 0; i < GPS_qr.size(); ++i){
        std::copy( GPS_qr[i].begin(), GPS_qr[i].end(), std::ostream_iterator<double>(out_gps, "\t"));
        out_gps << std::endl;
    }
    // //write number of feats into file
    //         os << feats.size()<<std::endl;
    // //write feats to file
    // for (size_t i=0; i < feats.size(); ++i )  {
    //         const openMVG::features::SIOPointFeature point = regions->Features()[i];
    //         os<< i<< "\t" <<point.x() <<"\t"<<point.y() << "\t" << point.scale() << "\t"<< point.orientation()<<std::endl;

    // }

}

int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy){
    int i, j, c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) &&
             (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
            c = !c;
    }
    return c;
}

///////////////////////////////////////////////////////////////////////compute function////////////////////////////////////////////////////////////////////////////////////
int compute(const int alg_idx, const std::string& GPS_path, const std::string& gaze_path, const std::string& qr_img_path){

    std::string rec_path =stlplus::folder_to_path( "../workspace");

    std::vector<std::string> paths;
    paths.push_back(rec_path);
    paths.push_back(GPS_path);
    paths.push_back(gaze_path);
    paths.push_back(qr_img_path);

    int topn = 2;
    float match_thr = 0.75;

    imretDataTypes::opt pOpt;
    pOpt.vvrepeats = 3;
    pOpt.vviter = 20;
    pOpt.vvsizeratio = 0.2;
    pOpt.vvmaxsize= 20000;
    pOpt.sadr_vv= "vv_kmeans";



    switch ( alg_idx ) {
    case 1:
        compute1( pOpt, paths,topn, match_thr );
        break;
    case 2:
        compute2( pOpt, paths,topn, match_thr );
        break;
    case 3:
        compute3( pOpt, paths,topn, match_thr );
        break;
    default:
        std::cout <<"no computing method is specified" << std::endl;
        break;
    }
}


int compute1(const imretDataTypes::opt& pOpt, std::vector<std::string>& paths, int topn, float match_thr ){

    std::string rec_path = paths[0];
    std::string GPS_path = paths[1];
    std::string gazes_path = paths[2];
    std::string qr_img_path =  paths[3];

    std::vector<std::vector<double> > matched_results;
    std::vector<std::vector<float> > transformed_labels;
    /////////////////offline parts/////////////////

    imretFuncs offlineParts;
    imretDataTypes::ENV env;
    offlineParts.getEnv(env);

    std::vector<std::string> img_list;
    std::vector<int> assgn, vw2im;


    if(stlplus::folder_empty(env.sim_path )){
        offlineParts.getFileList(img_list);
        offlineParts.simprep_gpu(img_list);
    }

    if(!stlplus::file_exists(pOpt.sadr_vv)){
        offlineParts.create_vv(pOpt);
    }

    if(!stlplus::folder_empty(env.sim_path)){
        img_list = stlplus::folder_wildcard (env.sim_path, "*.sift", false, true);
        std::sort(img_list.begin(), img_list.end(), compareNat);
        offlineParts.simquant(img_list,  pOpt,  assgn, vw2im);

    }


    std::string sim_path;
    offlineParts.get_sim_path(sim_path);


    std::vector<std::string> siftFiles = stlplus::folder_wildcard(sim_path, "*.sift", false, true);

    std::sort(siftFiles.begin(), siftFiles.end(), compareNat);

    imretDataTypes::data_qr data_qr_offline;
    imretDataTypes::data_vv data_vv_offline;
    offlineParts.load_ftrs_to_sim(data_qr_offline.assgn, data_qr_offline.vw2im, data_vv_offline.CN, data_vv_offline.N);

    Eigen::SparseMatrix<float> tfidf;
    offlineParts.tfidf_from_vvQR_fast(data_qr_offline, data_vv_offline, tfidf);


    //  Eigen::SparseMatrix<float> sim;
    //  offlineParts.simget(sim);


    ////////////load labels, gazes and gps////////////////////////////////////////

    std::vector <std::vector<double> > gazes,gps_assgn;
    getGazes(gazes_path,  gazes);
    std::string label_path_db = "../workspace/labels/labels_yy2.txt";
    std::string gps_proc_path = GPS_path;
    std::string gps_assgn_path = stlplus::folder_part(gps_proc_path) +"/"+ stlplus::basename_part(gps_proc_path)+"_assgn.txt";

    if(!stlplus::file_exists(gps_assgn_path)){
        getDBImID(gps_proc_path,  label_path_db,  gps_assgn);
    }else{
        std::ifstream in_gps(gps_assgn_path,std::ofstream::in);
        std::string line;
        while(std::getline(in_gps, line)){
            if(line !="" && line != "\n" && line != "\t"&& line!= " "){
                std::istringstream iss(line);
                std::vector<double> lineElems(7,0);
                int i;
                for(i = 0; i < lineElems.size(); ++i){
                    iss >> lineElems[i];
                }
                gps_assgn.push_back(lineElems);
            }
        }

    }
    ////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////


    //time peroids are measured in this part//


    /////////////////////////////online parts////////////////////////////

    std::string qr_sim_path = qr_img_path + "_sim/" ;
    std::string qr_imfmt = "png";
    //make sure the path ends with separator
    qr_img_path = stlplus::folder_append_separator(paths[3]);

    imretDataTypes::ENV qr_env{qr_img_path,qr_sim_path,qr_imfmt };

    imretFuncs onlineParts(rec_path, qr_env);
    int i,j,k,u;
    std::vector<std::string> qr_img_list,  img_to_proc;

    onlineParts.getFileList(qr_img_list);
    std::sort( qr_img_list.begin(), qr_img_list.end(), compareNat);


    /////////filter the frames with correctly synchronized gazes/////////

    for(i = 0; i < qr_img_list.size(); ++i){
        for(j = 0; j < gazes.size(); ++j){
            if(gazes[j][0] == (double)i){
                img_to_proc.push_back(qr_img_list[i]);
                break;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////

    ///////////check the difference between sift files and img files to decide if there are imgs whose sifts are not extracted//////////
    ///////////useful in real time application ///////////
    //
    // qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, "*.sift", false, true);
    // std::sort( qr_sift_list.begin(), qr_sift_list.end(), compareNat);
    // int diffSiftImg = qr_img_list.size() - qr_sift_list.size();
    // if(diffSiftImg){
    //         img_to_proc.resize(diffSiftImg);
    //         std::copy(qr_img_list.end() - diffSiftImg, qr_img_list.end(),img_to_proc.begin());
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int ii = 0;
    for(ii = 0; ii < img_to_proc.size(); ++ ii){


        gettimeofday(&starttime, NULL);


        std::vector<std::string> current_frame;
        current_frame.push_back(img_to_proc[ii]);
        onlineParts.simprep_gpu(current_frame);

        //simquant load sift files
        std::string sift_file_name = "dvw_" + stlplus::basename_part(img_to_proc[ii]) + ".sift";
        std::vector<std::string> qr_sift_list;
        qr_sift_list.push_back(sift_file_name);
        //                qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, "*.sift", false, true);
        //qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, sift_file_name, false, true);
        //std::sort(qr_sift_list.begin(), qr_sift_list.end(), compareNat);
        std::vector<int> qr_assgn, qr_vw2im;
        onlineParts.simquant(qr_sift_list, pOpt, qr_assgn, qr_vw2im);

        //similarity
        imretDataTypes::data_qr data_qr_online;
        imretDataTypes::data_vv data_vv_online;


        onlineParts.load_ftrs_to_sim(data_qr_online.assgn, data_qr_online.vw2im, data_vv_online.CN, data_vv_online.N);

        // std::copy(data_qr_online.assgn.begin(), data_qr_online.assgn.end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;


        Eigen::SparseMatrix<float>  tfidf_qr, simil;

        onlineParts.tfidf_from_vvQR_fast(data_qr_online, data_vv_online,tfidf_qr);

        simil = tfidf_qr.transpose()* tfidf;
        //convert simil to dense matrix
        MatrixXfr d_simil(simil);
        //load groundtruth data
        // std::string grnd_truth_fname = rec_path + "groundtruth/image.txt";
        // std::ifstream in_grnd_truth(grnd_truth_fname,std::ios::in);

        // std::vector<int> results;
        // int result;
        // while(in_grnd_truth>>result){
        //         results.push_back(result);
        // }
        // in_grnd_truth.close();

        // double correctVal = 0;
        // double countAll = 118;        //should be changed if the number of query images changes

        //////geometric verification/////

        std::cout << "geometric verification" << std::endl;



        //get image params
        openMVG::image::Image<unsigned char> testImage;
        img_list = stlplus::folder_wildcard(env.img_path, "*.png", false, true);
        std::sort(img_list.begin(), img_list.end(), compareNat);
        std::string imagePath = env.img_path + img_list[0];
        openMVG::image::ReadImage( imagePath.c_str(), &testImage);


        for(i = 0; i < d_simil.rows(); ++i){
            //load the descriptors of the query images
            MatrixXfr qr_descs_mat;
            flann::Matrix<Scalar> qr_descs_mat_flann;
            onlineParts.load_descs(qr_sim_path+qr_sift_list[i],qr_descs_mat);
            onlineParts.convert2flann(qr_descs_mat, qr_descs_mat_flann);

            std::unique_ptr< flann::Index<Metric> > qr_index;
            qr_index.reset(new flann::Index<Metric> (qr_descs_mat_flann, flann::KDTreeIndexParams(4)));
            qr_index->buildIndex();

            std::vector<float> qr_simil ((&(d_simil.row(i).data()[0])), (&(d_simil.row(i).data()[0]) + d_simil.cols()));
            std::vector<size_t> qr_simil_ind_sorted;
            std::vector<float> qr_simil_sorted;
            onlineParts.sort(qr_simil,qr_simil_sorted,qr_simil_ind_sorted);

            std::reverse(std::begin(qr_simil_ind_sorted), std::end(qr_simil_ind_sorted));

            //filter with gps data first by checking the neighbor buidlings of the building suggested by gps data
            std::vector<int> gps_suggested_idx;
            for(k = 0; k < gps_assgn.size(); ++k){
                if(gps_assgn[k].back()){

                    gps_suggested_idx.push_back((int)gps_assgn[k][1]);
                    if((gps_suggested_idx.size() > 1) && (gps_suggested_idx[gps_suggested_idx.size() - 2] == gps_suggested_idx.back())){
                        gps_suggested_idx.pop_back();
                    }

                }
            }

            std::vector<int> remain_topn;
            for(j = 0 ; j < topn; ++j){
                for(k = 0; k < gps_suggested_idx.size(); ++k){
                    if(qr_simil_ind_sorted[j] == gps_suggested_idx[k] ){
                        remain_topn.push_back(qr_simil_ind_sorted[j]);
                    }
                }
            }

            double dNfa = 999;
            int matchedImgId = -1;

            //                std::cout << gps_suggested_idx.size() << std::endl;
            if(!remain_topn.size()){
                if(gps_suggested_idx.size() < topn){
                    remain_topn.resize(gps_suggested_idx.size());
                    std::copy (gps_suggested_idx.begin(), gps_suggested_idx.end(), remain_topn.begin());
                }else{
                    remain_topn.resize(topn);
                    //  std::cout << gps_suggested_idx.size() << std::endl;
                    std::copy (gps_suggested_idx.begin(), gps_suggested_idx.begin() + topn, remain_topn.begin());
                }
            }

            std::vector<openMVG::Mat3> Hs;
            int bestMatchedIdx = 0;

            flann::SearchParams matching_search_params;
            matching_search_params.checks = 256;
            matching_search_params.cores = 0;


            for(j = 0; j < remain_topn.size(); ++j){
                MatrixXfr descs_mat;
                // std::cout << siftFiles[remain_topn[j]] << std::endl;
                flann::Matrix<Scalar> descs_mat_flann;
                //                std::cout <<i << " " << siftFiles[remain_topn[j]] << std::endl;
                offlineParts.load_descs(sim_path + siftFiles[remain_topn[j]],descs_mat);
                offlineParts.convert2flann(descs_mat, descs_mat_flann);
                std::unique_ptr< flann::Matrix<int> > indices;
                indices.reset(new flann::Matrix<int>(new int[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
                std::unique_ptr<flann::Matrix<DistanceType> > dists;
                dists.reset(new flann::Matrix<DistanceType>(new float[descs_mat_flann.rows*2], descs_mat_flann.rows,2));

                qr_index->knnSearch(descs_mat_flann, *indices, *dists, 2, matching_search_params);

                Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> indices_mat(indices->ptr(), indices->rows,indices->cols);

                MatrixXfr tmp1(descs_mat.rows(),128);
                MatrixXfr tmp2(descs_mat.rows(),128);

                for(k = 0; k < descs_mat.rows(); ++k){
                    MatrixXfr tmp(1,128);
                    tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,0));
                    tmp1.row(k)= tmp.array().square();
                    tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,1));
                    tmp2.row(k) = tmp.array().square();
                }
                MatrixXfr tmp1_sum;
                MatrixXfr tmp2_sum;
                Eigen::Matrix<bool,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gm;
                tmp1_sum = tmp1.rowwise().sum().eval();
                tmp2_sum = tmp2.rowwise().sum().eval() * match_thr* match_thr;
                gm = tmp1_sum.array() < tmp2_sum.array();

                std::vector<int> gm_ind;

                for(k =0; k<gm.rows(); ++k){
                    if(gm(k,0)){
                        gm_ind.push_back(k);
                    }
                }

                std::sort(gm_ind.begin(), gm_ind.end());
                std::vector<int> sel;
                std::vector<int> indices_unique;
                indices_unique.push_back(indices_mat(gm_ind[0],0));
                sel.push_back(0);
                for(k = 1; k < gm_ind.size(); ++k){
                    if(std::find(indices_unique.begin(), indices_unique.end(),indices_mat(gm_ind[k],0))==indices_unique.end()){
                        indices_unique.push_back(indices_mat(gm_ind[k],0));
                        sel.push_back(k);
                    }
                }

                MatrixXfr qr_feats_mat,feats_mat, tc(sel.size(),4), tc_qr(sel.size(),4);
                offlineParts.load_feats(sim_path + siftFiles[remain_topn[j]],feats_mat);
                onlineParts.load_feats(qr_sim_path + qr_sift_list[i], qr_feats_mat);

                int n = 0,m = 0;
                for(k =0; k < sel.size(); ++k){
                    int idx_tc = gm_ind[sel[k]];
                    int idx_tc_qr = indices_mat(gm_ind[k]);
                    if(idx_tc<=feats_mat.cols()){
                        tc.row(n) = feats_mat.row(idx_tc);
                        ++n;
                    }

                    if(idx_tc_qr<=qr_feats_mat.rows()){
                        tc_qr.row(m) = qr_feats_mat.row(indices_mat(gm_ind[sel[k]],1));
                        ++m;
                    }
                }

                if(n-1<sel.size()){
                    tc.resize(n-1,4);
                }

                if(m-1<sel.size()){
                    tc_qr.resize (m-1,4);
                }

                ////////////////////ACRANSAC/////////////////////////////

                std::vector<size_t> vec_inliers;
                typedef openMVG::robust::ACKernelAdaptor<
                        openMVG::homography::kernel::FourPointSolver,
                        openMVG::homography::kernel::AsymmetricError,
                        openMVG::UnnormalizerI,
                        openMVG::Mat3> KernelType;

                openMVG::Mat tc_openMVG;
                openMVG::Mat tc_qr_openMVG;

                if(tc.rows()==tc_qr.rows()){
                    tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                    tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();

                }else{
                    if(tc.rows() < tc_qr.rows()){
                        tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                        tc_qr_openMVG = tc_qr.block(0,0,tc.rows(),2).eval().transpose().cast<double>();
                    }else{
                        tc_openMVG = tc.block(0,0,tc_qr.rows(),2).eval().transpose().cast<double>();
                        tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();
                    }
                }

                KernelType kernel(
                            tc_openMVG, testImage.Width(), testImage.Height(),
                            tc_qr_openMVG,testImage.Width(), testImage.Height(),
                            false); // configure as point to point error model.

                openMVG::Mat3 H;
                std::pair<double,double> ACRansacOut = ACRANSAC(kernel, vec_inliers, 100, &H,
                                                                std::numeric_limits<double>::infinity(),
                                                                false);

                Hs.push_back(H);

                if(ACRansacOut.first <= dNfa){
                    dNfa = ACRansacOut.first;
                    matchedImgId = remain_topn[j];
                    bestMatchedIdx = j;
                }




                //////////////////////////////////USAC/////////////////////////////////////////
                // HomogEstimator* homog = new HomogEstimator;

                // // initialize the USAC parameters, either from a config file, or from your application
                // homog->initParamsUSAC(cfg);

                // // get input data points/prosac ordering data (if required)
                // // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

                // // set up the estimation problem
                // homog->initDataUSAC(cfg);
                // homog->initProblem(cfg, &point_data);

                // // solve
                // if (!homog->solve()){
                //         return(EXIT_FAILURE);
                // }

                // // do stuff

                // // cleanup
                // homog->cleanupProblem();
                // delete homog;


            }


            openMVG::Mat3 bestMatched_H = Hs[bestMatchedIdx];

            /////////////gaze matching//////////////

            std::vector<double> matched_result_element(4,0);
            matched_result_element[0] = gazes[ii][0];
            matched_result_element[2] = matchedImgId;

            for(j = 0; j < gps_assgn.size(); ++j ){
                if ((double)matchedImgId == gps_assgn[j][1]){
                    //project frame to gaze with homography

                    openMVG::Mat verts(3,4);
                    verts<< gps_assgn[j][2],gps_assgn[j][2], gps_assgn[j][4],gps_assgn[j][4],
                            gps_assgn[j][5],gps_assgn[j][3],gps_assgn[j][3], gps_assgn[j][5],
                            1,1,1,1;

                    verts = bestMatched_H.inverse()*verts;
                    // openMVG::Mat gazes_mat(3,1);
                    // gazes_mat << gazes[i][1],
                    //         gazes[i][2],
                    //         1;
                    //gazes_mat = bestMatched_H*gazes_mat;

                    float vertx[4] = {(float)(verts(0,0)/verts(2,0)), (float)(verts(0,1)/verts(2,1)),(float)(verts(0,2)/verts(2,2)), (float)(verts(0,3)/verts(2,3))};
                    float verty[4] = {(float)(verts(1,0)/verts(2,0)),  (float)(verts(1,1)/verts(2,1)),(float)(verts(1,2)/verts(2,2)), (float)(verts(1,3)/verts(2,3))};
                    float testx = gazes[ii][1];
                    float testy = gazes[ii][2];
                    //                                 float testx = (float)(gazes_mat(0,0)/gazes_mat(2,0));
                    //                                 float testy = (float)(gazes_mat(1,0)/gazes_mat(2,0));

                    //keep the transformed bounding box in memory
                    std::vector<float> transformed_label_x(std::begin(vertx), std::end(vertx));
                    std::vector<float> transformed_label_y(std::begin(verty), std::end(verty));
                    transformed_label_x.insert(transformed_label_x.end(),transformed_label_y.begin(), transformed_label_y.end());
                    transformed_labels.push_back(transformed_label_x);



                    if( pnpoly(4, vertx, verty,  testx,testy) ){
                        matched_result_element[1] = gps_assgn[j][0];
                        //record if the homography is useful
                        matched_result_element[3] = 1;
                    }
                }
            }
            matched_results.push_back(matched_result_element);
            // std::cout << siftFiles[matchedImgId] << std::endl;
            // for( j = 0; j < results.size(); ++j){
            //         if(matchedImgId == j){
            //                 if(results[j] == 1){
            //                        correctVal++;
            //                        break;
            //                 }
            //                        //countAll++;
            //         }

            // }

            qr_index.reset();

        } //to loop through the similarity matrix row by row, in this case, there is only one row because there is only one frame


        gettimeofday(&endtime, NULL);

        double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;
        std::cout << delta << std::endl;




        //delete ftrs_to_sim file
        std::string ftrs_to_sim_path = qr_sim_path + "ftrs_to_sim";
        stlplus::file_delete (ftrs_to_sim_path);

    }// to feature-matching one frame by another


    //write out the final results
    // std::cout << "writing out final results" << std::endl;
    // std::string final_results_path = "../workspace/results/results.txt";
    // std::ofstream out_results (final_results_path, std::ofstream::out);
    // for (size_t i=0; i < matched_results.size(); ++i )  {
    //         std::copy( matched_results[i].begin(), matched_results[i].end(), std::ostream_iterator<double>(out_results, "\t"));
    //         out_results << std::endl;
    // }

    return 0;

}//algorithm 3





int compute2(const imretDataTypes::opt& pOpt, std::vector<std::string>& paths, int topn, float match_thr ){
    std::string rec_path = paths[0];
    std::string GPS_path = paths[1];
    std::string gazes_path = paths[2];
    std::string qr_img_path =  paths[3];

    std::vector<std::vector<double> > matched_results;
    std::vector<std::vector<float> > transformed_labels;
    /////////////////offline parts/////////////////

    imretFuncs offlineParts;
    imretDataTypes::ENV env;
    offlineParts.getEnv(env);

    std::vector<std::string> img_list;
    std::vector<int> assgn, vw2im;


    if(stlplus::folder_empty(env.sim_path )){
        offlineParts.getFileList(img_list);
        offlineParts.simprep_gpu(img_list);
    }

    if(!stlplus::file_exists(pOpt.sadr_vv)){
        offlineParts.create_vv(pOpt);
    }

    if(!stlplus::folder_empty(env.sim_path)){
        img_list = stlplus::folder_wildcard (env.sim_path, "*.sift", false, true);
        std::sort(img_list.begin(), img_list.end(), compareNat);
        offlineParts.simquant(img_list,  pOpt,  assgn, vw2im);

    }


    std::string sim_path;
    offlineParts.get_sim_path(sim_path);


    std::vector<std::string> siftFiles = stlplus::folder_wildcard(sim_path, "*.sift", false, true);

    std::sort(siftFiles.begin(), siftFiles.end(), compareNat);

    imretDataTypes::data_qr data_qr_offline;
    imretDataTypes::data_vv data_vv_offline;
    offlineParts.load_ftrs_to_sim(data_qr_offline.assgn, data_qr_offline.vw2im, data_vv_offline.CN, data_vv_offline.N);

    Eigen::SparseMatrix<float> tfidf;
    offlineParts.tfidf_from_vvQR_fast(data_qr_offline, data_vv_offline, tfidf);


    //  Eigen::SparseMatrix<float> sim;
    //  offlineParts.simget(sim);


    ////////////load labels, gazes and gps////////////////////////////////////////

    std::vector <std::vector<double> > gazes,gps_assgn;
    getGazes(gazes_path,  gazes);
    std::string label_path_db = "../workspace/labels/labels_yy2.txt";
    std::string gps_proc_path = GPS_path;
    std::string gps_assgn_path = stlplus::folder_part(gps_proc_path) +"/"+ stlplus::basename_part(gps_proc_path)+"_assgn.txt";

    if(!stlplus::file_exists(gps_assgn_path)){
        getDBImID(gps_proc_path,  label_path_db,  gps_assgn);
    }else{
        std::ifstream in_gps(gps_assgn_path,std::ofstream::in);
        std::string line;
        while(std::getline(in_gps, line)){
            if(line !="" && line != "\n" && line != "\t"&& line!= " "){
                std::istringstream iss(line);
                std::vector<double> lineElems(7,0);
                int i;
                for(i = 0; i < lineElems.size(); ++i){
                    iss >> lineElems[i];
                }
                gps_assgn.push_back(lineElems);
            }
        }

    }
    ////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////


    //time peroids are measured in this part//


    /////////////////////////////online parts////////////////////////////

    std::string qr_sim_path = qr_img_path + "_sim/" ;
    std::string qr_imfmt = "png";
    //make sure the path ends with separator
    qr_img_path = stlplus::folder_append_separator(paths[3]);

    imretDataTypes::ENV qr_env{qr_img_path,qr_sim_path,qr_imfmt };

    imretFuncs onlineParts(rec_path, qr_env);
    int i,j,k,u;
    std::vector<std::string> qr_img_list,  img_to_proc;

    onlineParts.getFileList(qr_img_list);
    std::sort( qr_img_list.begin(), qr_img_list.end(), compareNat);


    /////////filter the frames with correctly synchronized gazes/////////

    for(i = 0; i < qr_img_list.size(); ++i){
        for(j = 0; j < gazes.size(); ++j){
            if(gazes[j][0] == (double)i){
                img_to_proc.push_back(qr_img_list[i]);
                break;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////

    ///////////check the difference between sift files and img files to decide if there are imgs whose sifts are not extracted//////////
    ///////////useful in real time application ///////////
    //
    // qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, "*.sift", false, true);
    // std::sort( qr_sift_list.begin(), qr_sift_list.end(), compareNat);
    // int diffSiftImg = qr_img_list.size() - qr_sift_list.size();
    // if(diffSiftImg){
    //         img_to_proc.resize(diffSiftImg);
    //         std::copy(qr_img_list.end() - diffSiftImg, qr_img_list.end(),img_to_proc.begin());
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int ii = 0;
    for(ii = 0; ii < img_to_proc.size(); ++ ii){

        std::vector<std::string> current_frame;
        current_frame.push_back(img_to_proc[ii]);
        onlineParts.simprep_gpu(current_frame);
        //simquant load sift files
        std::string sift_file_name = "dvw_" + stlplus::basename_part(img_to_proc[ii]) + ".sift";
        std::vector<std::string> qr_sift_list;
        qr_sift_list.push_back(sift_file_name);

        if(ii>0){

            //////////////////////////algorithm 2: use former matched image ///////////////////////////////////////
            /////////////////////////always try algorithm 2 before 3 and abandon it when nothing is matched////////
            std::cout << "algorithm 2 is used" << std::endl;


            gettimeofday(&starttime, NULL);


            MatrixXfr qr_descs_mat;
            flann::Matrix<Scalar> qr_descs_mat_flann;
            onlineParts.load_descs(qr_sim_path+qr_sift_list[0],qr_descs_mat);
            onlineParts.convert2flann(qr_descs_mat, qr_descs_mat_flann);



            std::unique_ptr< flann::Index<Metric> > qr_index;
            qr_index.reset(new flann::Index<Metric> (qr_descs_mat_flann, flann::KDTreeIndexParams(4)));
            qr_index->buildIndex();

            int suggested_img_id =(int) matched_results[ii-1][2];



            openMVG::image::Image<unsigned char> testImage;
            img_list = stlplus::folder_wildcard(env.img_path, "*.png", false, true);
            std::sort(img_list.begin(), img_list.end(), compareNat);
            std::string imagePath = env.img_path + img_list[0];
            openMVG::image::ReadImage( imagePath.c_str(), &testImage);





            MatrixXfr descs_mat;
            flann::Matrix<Scalar> descs_mat_flann;
            offlineParts.load_descs(sim_path + siftFiles[suggested_img_id],descs_mat);
            offlineParts.convert2flann(descs_mat, descs_mat_flann);
            std::unique_ptr< flann::Matrix<int> > indices;
            indices.reset(new flann::Matrix<int>(new int[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
            std::unique_ptr<flann::Matrix<DistanceType> > dists;
            dists.reset(new flann::Matrix<DistanceType>(new float[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
            flann::SearchParams matching_search_params;
            matching_search_params.checks = 256;
            matching_search_params.cores = 0;
            qr_index->knnSearch(descs_mat_flann, *indices, *dists, 2, matching_search_params);




            Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> indices_mat(indices->ptr(), indices->rows,indices->cols);

            MatrixXfr tmp1(descs_mat.rows(),128);
            MatrixXfr tmp2(descs_mat.rows(),128);

            for(k = 0; k < descs_mat.rows(); ++k){
                MatrixXfr tmp(1,128);
                tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,0));
                tmp1.row(k)= tmp.array().square();
                tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,1));
                tmp2.row(k) = tmp.array().square();
            }
            MatrixXfr tmp1_sum;
            MatrixXfr tmp2_sum;
            Eigen::Matrix<bool,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gm;
            tmp1_sum = tmp1.rowwise().sum().eval();
            tmp2_sum = tmp2.rowwise().sum().eval() * match_thr* match_thr;
            gm = tmp1_sum.array() < tmp2_sum.array();

            std::vector<int> gm_ind;

            for(k =0; k<gm.rows(); ++k){
                if(gm(k,0)){
                    gm_ind.push_back(k);
                }
            }

            std::sort(gm_ind.begin(), gm_ind.end());
            std::vector<int> sel;
            std::vector<int> indices_unique;
            indices_unique.push_back(indices_mat(gm_ind[0],0));
            sel.push_back(0);
            for(k = 1; k < gm_ind.size(); ++k){
                if(std::find(indices_unique.begin(), indices_unique.end(),indices_mat(gm_ind[k],0))==indices_unique.end()){
                    indices_unique.push_back(indices_mat(gm_ind[k],0));
                    sel.push_back(k);
                }
            }


            MatrixXfr qr_feats_mat,feats_mat, tc(sel.size(),4), tc_qr(sel.size(),4);
            offlineParts.load_feats(sim_path + siftFiles[suggested_img_id],feats_mat);
            onlineParts.load_feats(qr_sim_path + qr_sift_list[0], qr_feats_mat);

            int n = 0,m = 0;
            for(k =0; k < sel.size(); ++k){
                int idx_tc = gm_ind[sel[k]];
                int idx_tc_qr = indices_mat(gm_ind[k]);
                if(idx_tc<=feats_mat.cols()){
                    tc.row(n) = feats_mat.row(idx_tc);
                    ++n;
                }

                if(idx_tc_qr<=qr_feats_mat.rows()){
                    tc_qr.row(m) = qr_feats_mat.row(indices_mat(gm_ind[sel[k]],1));
                    ++m;
                }
            }

            if(n-1<sel.size()){
                tc.resize(n-1,4);
            }

            if(m-1<sel.size()){
                tc_qr.resize (m-1,4);
            }


            ////////////////////ACRANSAC/////////////////////////////

            std::vector<size_t> vec_inliers;
            typedef openMVG::robust::ACKernelAdaptor<
                    openMVG::homography::kernel::FourPointSolver,
                    openMVG::homography::kernel::AsymmetricError,
                    openMVG::UnnormalizerI,
                    openMVG::Mat3> KernelType;

            openMVG::Mat tc_openMVG;
            openMVG::Mat tc_qr_openMVG;

            if(tc.rows()==tc_qr.rows()){
                tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();

            }else{
                if(tc.rows() < tc_qr.rows()){
                    tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                    tc_qr_openMVG = tc_qr.block(0,0,tc.rows(),2).eval().transpose().cast<double>();
                }else{
                    tc_openMVG = tc.block(0,0,tc_qr.rows(),2).eval().transpose().cast<double>();
                    tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();
                }
            }

            KernelType kernel(
                        tc_openMVG, testImage.Width(), testImage.Height(),
                        tc_qr_openMVG,testImage.Width(), testImage.Height(),
                        false); // configure as point to point error model.

            openMVG::Mat3 H;
            std::pair<double,double> ACRansacOut = ACRANSAC(kernel, vec_inliers, 100, &H,
                                                            std::numeric_limits<double>::infinity(),
                                                            false);








            // std::cout << "qr " << qr_sift_list[i] << std::endl;
            // std::cout <<"db " << siftFiles[qr_simil_ind_sorted[0]] <<" " <<siftFiles[qr_simil_ind_sorted[1]] <<" "<<siftFiles[qr_simil_ind_sorted[2]]<< std::endl;
            // std::cout << "recommended " << siftFiles[matchedImgId] << std::endl;
            // std::cout << std::endl;


            //////////////////////////////////USAC/////////////////////////////////////////
            // HomogEstimator* homog = new HomogEstimator;

            // // initialize the USAC parameters, either from a config file, or from your application
            // homog->initParamsUSAC(cfg);

            // // get input data points/prosac ordering data (if required)
            // // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

            // // set up the estimation problem
            // homog->initDataUSAC(cfg);
            // homog->initProblem(cfg, &point_data);

            // // solve
            // if (!homog->solve()){
            //         return(EXIT_FAILURE);
            // }

            // // do stuff

            // // cleanup
            // homog->cleanupProblem();
            // delete homog;






            /////////////gaze matching//////////////

            std::vector<double> matched_result_element(4,0);
            matched_result_element[0] = gazes[ii][0];
            matched_result_element[2] = suggested_img_id;

            for(j = 0; j < gps_assgn.size(); ++j ){
                if ((double)suggested_img_id == gps_assgn[j][1]){
                    //project frame to gaze with homography

                    openMVG::Mat verts(3,4);
                    verts<< gps_assgn[j][2],gps_assgn[j][2], gps_assgn[j][4],gps_assgn[j][4],
                            gps_assgn[j][5],gps_assgn[j][3],gps_assgn[j][3], gps_assgn[j][5],
                            1,1,1,1;

                    verts = H.inverse()*verts;
                    // openMVG::Mat gazes_mat(3,1);
                    // gazes_mat << gazes[i][1],
                    //         gazes[i][2],
                    //         1;
                    //gazes_mat = bestMatched_H*gazes_mat;

                    float vertx[4] = {(float)(verts(0,0)/verts(2,0)), (float)(verts(0,1)/verts(2,1)),(float)(verts(0,2)/verts(2,2)), (float)(verts(0,3)/verts(2,3))};
                    float verty[4] = {(float)(verts(1,0)/verts(2,0)),  (float)(verts(1,1)/verts(2,1)),(float)(verts(1,2)/verts(2,2)), (float)(verts(1,3)/verts(2,3))};
                    float testx = gazes[ii][1];
                    float testy = gazes[ii][2];
                    //                                 float testx = (float)(gazes_mat(0,0)/gazes_mat(2,0));
                    //                                 float testy = (float)(gazes_mat(1,0)/gazes_mat(2,0));

                    //keep the transformed bounding box in memory
                    std::vector<float> transformed_label_x(std::begin(vertx), std::end(vertx));
                    std::vector<float> transformed_label_y(std::begin(verty), std::end(verty));
                    transformed_label_x.insert(transformed_label_x.end(),transformed_label_y.begin(), transformed_label_y.end());
                    transformed_labels.push_back(transformed_label_x);



                    if( pnpoly(4, vertx, verty,  testx,testy) ){
                        matched_result_element[1] = gps_assgn[j][0];
                        //record if the homography is useful
                        matched_result_element[3] = 1;
                    }
                }
            }

            matched_results.push_back(matched_result_element);
            gettimeofday(&endtime, NULL);
            double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;


            std::cout <<delta << std::endl;
            qr_index.reset();
        }

        if((ii==0)|| (!matched_results[ii][3])){
            //////////////////////////
            ///
            ///
            /// : query the database///////////////////////////////
            std::cout << "algorithm 3 is used" << std::endl;



            gettimeofday(&starttime, NULL);


            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            //                qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, "*.sift", false, true);
            //qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, sift_file_name, false, true);
            //std::sort(qr_sift_list.begin(), qr_sift_list.end(), compareNat);
            std::vector<int> qr_assgn, qr_vw2im;
            onlineParts.simquant(qr_sift_list, pOpt, qr_assgn, qr_vw2im);

            //similarity
            imretDataTypes::data_qr data_qr_online;
            imretDataTypes::data_vv data_vv_online;


            onlineParts.load_ftrs_to_sim(data_qr_online.assgn, data_qr_online.vw2im, data_vv_online.CN, data_vv_online.N);

            // std::copy(data_qr_online.assgn.begin(), data_qr_online.assgn.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << std::endl;


            Eigen::SparseMatrix<float>  tfidf_qr, simil;

            onlineParts.tfidf_from_vvQR_fast(data_qr_online, data_vv_online,tfidf_qr);

            simil = tfidf_qr.transpose()* tfidf;
            //convert simil to dense matrix
            MatrixXfr d_simil(simil);
            //load groundtruth data
            // std::string grnd_truth_fname = rec_path + "groundtruth/image.txt";
            // std::ifstream in_grnd_truth(grnd_truth_fname,std::ios::in);

            // std::vector<int> results;
            // int result;
            // while(in_grnd_truth>>result){
            //         results.push_back(result);
            // }
            // in_grnd_truth.close();

            // double correctVal = 0;
            // double countAll = 118;        //should be changed if the number of query images changes

            //////geometric verification/////

            std::cout << "geometric verification" << std::endl;



            //get image params
            openMVG::image::Image<unsigned char> testImage;
            img_list = stlplus::folder_wildcard(env.img_path, "*.png", false, true);
            std::sort(img_list.begin(), img_list.end(), compareNat);
            std::string imagePath = env.img_path + img_list[0];
            openMVG::image::ReadImage( imagePath.c_str(), &testImage);




            for(i = 0; i < d_simil.rows(); ++i){
                //load the descriptors of the query images
                MatrixXfr qr_descs_mat;
                flann::Matrix<Scalar> qr_descs_mat_flann;
                onlineParts.load_descs(qr_sim_path+qr_sift_list[i],qr_descs_mat);
                onlineParts.convert2flann(qr_descs_mat, qr_descs_mat_flann);

                std::unique_ptr< flann::Index<Metric> > qr_index;
                qr_index.reset(new flann::Index<Metric> (qr_descs_mat_flann, flann::KDTreeIndexParams(4)));
                qr_index->buildIndex();

                std::vector<float> qr_simil ((&(d_simil.row(i).data()[0])), (&(d_simil.row(i).data()[0]) + d_simil.cols()));
                std::vector<size_t> qr_simil_ind_sorted;
                std::vector<float> qr_simil_sorted;
                onlineParts.sort(qr_simil,qr_simil_sorted,qr_simil_ind_sorted);

                std::reverse(std::begin(qr_simil_ind_sorted), std::end(qr_simil_ind_sorted));

                //filter with gps data first by checking the neighbor buidlings of the building suggested by gps data
                std::vector<int> gps_suggested_idx;
                for(k = 0; k < gps_assgn.size(); ++k){
                    if(gps_assgn[k].back()){

                        gps_suggested_idx.push_back((int)gps_assgn[k][1]);
                        if((gps_suggested_idx.size() > 1) && (gps_suggested_idx[gps_suggested_idx.size() - 2] == gps_suggested_idx.back())){
                            gps_suggested_idx.pop_back();
                        }

                    }
                }

                std::vector<int> remain_topn;
                for(j = 0 ; j < topn; ++j){
                    for(k = 0; k < gps_suggested_idx.size(); ++k){
                        if(qr_simil_ind_sorted[j] == gps_suggested_idx[k] ){
                            remain_topn.push_back(qr_simil_ind_sorted[j]);
                        }
                    }
                }

                double dNfa = 999;
                int matchedImgId = -1;

                //                std::cout << gps_suggested_idx.size() << std::endl;
                if(!remain_topn.size()){
                    if(gps_suggested_idx.size() < topn){
                        remain_topn.resize(gps_suggested_idx.size());
                        std::copy (gps_suggested_idx.begin(), gps_suggested_idx.end(), remain_topn.begin());
                    }else{
                        remain_topn.resize(topn);
                        //  std::cout << gps_suggested_idx.size() << std::endl;
                        std::copy (gps_suggested_idx.begin(), gps_suggested_idx.begin() + topn, remain_topn.begin());
                    }
                }

                std::vector<openMVG::Mat3> Hs;
                int bestMatchedIdx = 0;
                flann::SearchParams matching_search_params;
                matching_search_params.checks = 256;
                matching_search_params.cores = 0;
                for(j = 0; j < remain_topn.size(); ++j){
                    MatrixXfr descs_mat;
                    // std::cout << siftFiles[remain_topn[j]] << std::endl;
                    flann::Matrix<Scalar> descs_mat_flann;
                    //                std::cout <<i << " " << siftFiles[remain_topn[j]] << std::endl;
                    offlineParts.load_descs(sim_path + siftFiles[remain_topn[j]],descs_mat);
                    offlineParts.convert2flann(descs_mat, descs_mat_flann);
                    std::unique_ptr< flann::Matrix<int> > indices;
                    indices.reset(new flann::Matrix<int>(new int[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
                    std::unique_ptr<flann::Matrix<DistanceType> > dists;
                    dists.reset(new flann::Matrix<DistanceType>(new float[descs_mat_flann.rows*2], descs_mat_flann.rows,2));


                    qr_index->knnSearch(descs_mat_flann, *indices, *dists, 2, matching_search_params);



                    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> indices_mat(indices->ptr(), indices->rows,indices->cols);

                    MatrixXfr tmp1(descs_mat.rows(),128);
                    MatrixXfr tmp2(descs_mat.rows(),128);

                    for(k = 0; k < descs_mat.rows(); ++k){
                        MatrixXfr tmp(1,128);
                        tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,0));
                        tmp1.row(k)= tmp.array().square();
                        tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,1));
                        tmp2.row(k) = tmp.array().square();
                    }
                    MatrixXfr tmp1_sum;
                    MatrixXfr tmp2_sum;
                    Eigen::Matrix<bool,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gm;
                    tmp1_sum = tmp1.rowwise().sum().eval();
                    tmp2_sum = tmp2.rowwise().sum().eval() * match_thr* match_thr;
                    gm = tmp1_sum.array() < tmp2_sum.array();

                    std::vector<int> gm_ind;

                    for(k =0; k<gm.rows(); ++k){
                        if(gm(k,0)){
                            gm_ind.push_back(k);
                        }
                    }

                    std::sort(gm_ind.begin(), gm_ind.end());
                    std::vector<int> sel;
                    std::vector<int> indices_unique;
                    indices_unique.push_back(indices_mat(gm_ind[0],0));
                    sel.push_back(0);
                    for(k = 1; k < gm_ind.size(); ++k){
                        if(std::find(indices_unique.begin(), indices_unique.end(),indices_mat(gm_ind[k],0))==indices_unique.end()){
                            indices_unique.push_back(indices_mat(gm_ind[k],0));
                            sel.push_back(k);
                        }
                    }

                    MatrixXfr qr_feats_mat,feats_mat, tc(sel.size(),4), tc_qr(sel.size(),4);
                    offlineParts.load_feats(sim_path + siftFiles[remain_topn[j]],feats_mat);
                    onlineParts.load_feats(qr_sim_path + qr_sift_list[i], qr_feats_mat);

                    int n = 0,m = 0;
                    for(k =0; k < sel.size(); ++k){
                        int idx_tc = gm_ind[sel[k]];
                        int idx_tc_qr = indices_mat(gm_ind[k]);
                        if(idx_tc<=feats_mat.cols()){
                            tc.row(n) = feats_mat.row(idx_tc);
                            ++n;
                        }

                        if(idx_tc_qr<=qr_feats_mat.rows()){
                            tc_qr.row(m) = qr_feats_mat.row(indices_mat(gm_ind[sel[k]],1));
                            ++m;
                        }
                    }

                    if(n-1<sel.size()){
                        tc.resize(n-1,4);
                    }

                    if(m-1<sel.size()){
                        tc_qr.resize (m-1,4);
                    }

                    ////////////////////ACRANSAC/////////////////////////////

                    std::vector<size_t> vec_inliers;
                    typedef openMVG::robust::ACKernelAdaptor<
                            openMVG::homography::kernel::FourPointSolver,
                            openMVG::homography::kernel::AsymmetricError,
                            openMVG::UnnormalizerI,
                            openMVG::Mat3> KernelType;

                    openMVG::Mat tc_openMVG;
                    openMVG::Mat tc_qr_openMVG;

                    if(tc.rows()==tc_qr.rows()){
                        tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                        tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();

                    }else{
                        if(tc.rows() < tc_qr.rows()){
                            tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                            tc_qr_openMVG = tc_qr.block(0,0,tc.rows(),2).eval().transpose().cast<double>();
                        }else{
                            tc_openMVG = tc.block(0,0,tc_qr.rows(),2).eval().transpose().cast<double>();
                            tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();
                        }
                    }

                    KernelType kernel(
                                tc_openMVG, testImage.Width(), testImage.Height(),
                                tc_qr_openMVG,testImage.Width(), testImage.Height(),
                                false); // configure as point to point error model.

                    openMVG::Mat3 H;
                    std::pair<double,double> ACRansacOut = ACRANSAC(kernel, vec_inliers, 100, &H,
                                                                    std::numeric_limits<double>::infinity(),
                                                                    false);

                    Hs.push_back(H);

                    if(ACRansacOut.first <= dNfa){
                        dNfa = ACRansacOut.first;
                        matchedImgId = remain_topn[j];
                        bestMatchedIdx = j;
                    }



                    // std::cout << "qr " << qr_sift_list[i] << std::endl;
                    // std::cout <<"db " << siftFiles[qr_simil_ind_sorted[0]] <<" " <<siftFiles[qr_simil_ind_sorted[1]] <<" "<<siftFiles[qr_simil_ind_sorted[2]]<< std::endl;
                    // std::cout << "recommended " << siftFiles[matchedImgId] << std::endl;
                    // std::cout << std::endl;


                    //////////////////////////////////USAC/////////////////////////////////////////
                    // HomogEstimator* homog = new HomogEstimator;

                    // // initialize the USAC parameters, either from a config file, or from your application
                    // homog->initParamsUSAC(cfg);

                    // // get input data points/prosac ordering data (if required)
                    // // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

                    // // set up the estimation problem
                    // homog->initDataUSAC(cfg);
                    // homog->initProblem(cfg, &point_data);

                    // // solve
                    // if (!homog->solve()){
                    //         return(EXIT_FAILURE);
                    // }

                    // // do stuff

                    // // cleanup
                    // homog->cleanupProblem();
                    // delete homog;


                }


                openMVG::Mat3 bestMatched_H = Hs[bestMatchedIdx];

                /////////////gaze matching//////////////

                std::vector<double> matched_result_element(4,0);
                matched_result_element[0] = gazes[ii][0];
                matched_result_element[2] = matchedImgId;


                for(j = 0; j < gps_assgn.size(); ++j ){
                    if ((double)matchedImgId == gps_assgn[j][1]){
                        //project frame to gaze with homography

                        openMVG::Mat verts(3,4);
                        verts<< gps_assgn[j][2],gps_assgn[j][2], gps_assgn[j][4],gps_assgn[j][4],
                                gps_assgn[j][5],gps_assgn[j][3],gps_assgn[j][3], gps_assgn[j][5],
                                1,1,1,1;

                        verts = bestMatched_H.inverse()*verts;
                        // openMVG::Mat gazes_mat(3,1);
                        // gazes_mat << gazes[i][1],
                        //         gazes[i][2],
                        //         1;
                        //gazes_mat = bestMatched_H*gazes_mat;

                        float vertx[4] = {(float)(verts(0,0)/verts(2,0)), (float)(verts(0,1)/verts(2,1)),(float)(verts(0,2)/verts(2,2)), (float)(verts(0,3)/verts(2,3))};
                        float verty[4] = {(float)(verts(1,0)/verts(2,0)),  (float)(verts(1,1)/verts(2,1)),(float)(verts(1,2)/verts(2,2)), (float)(verts(1,3)/verts(2,3))};
                        float testx = gazes[ii][1];
                        float testy = gazes[ii][2];
                        //                                 float testx = (float)(gazes_mat(0,0)/gazes_mat(2,0));
                        //                                 float testy = (float)(gazes_mat(1,0)/gazes_mat(2,0));

                        //keep the transformed bounding box in memory
                        std::vector<float> transformed_label_x(std::begin(vertx), std::end(vertx));
                        std::vector<float> transformed_label_y(std::begin(verty), std::end(verty));
                        transformed_label_x.insert(transformed_label_x.end(),transformed_label_y.begin(), transformed_label_y.end());
                        transformed_labels.push_back(transformed_label_x);



                        if( pnpoly(4, vertx, verty,  testx,testy) ){
                            matched_result_element[1] = gps_assgn[j][0];
                            //record if the homography is useful
                            matched_result_element[3] = 1;
                        }
                    }
                }
                matched_results.push_back(matched_result_element);
                // std::cout << siftFiles[matchedImgId] << std::endl;
                // for( j = 0; j < results.size(); ++j){
                //         if(matchedImgId == j){
                //                 if(results[j] == 1){
                //                        correctVal++;
                //                        break;
                //                 }
                //                        //countAll++;
                //         }

                // }

                qr_index.reset();


            } //to loop through the similarity matrix row by row, in this case, there is only one row because there is only one frame

            gettimeofday(&endtime, NULL);
            double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;


            std::cout << delta<< std::endl;

        }//algorithm 3

        //delete ftrs_to_sim file
        std::string ftrs_to_sim_path = qr_sim_path + "ftrs_to_sim";
        stlplus::file_delete (ftrs_to_sim_path);


    }// to feature-matching one frame by another



    //write out the final results
    // std::cout << "writing out final results" << std::endl;
    // std::string final_results_path = "../workspace/results/results.txt";
    // std::ofstream out_results (final_results_path, std::ofstream::out);
    // for (size_t i=0; i < matched_results.size(); ++i )  {
    //         std::copy( matched_results[i].begin(), matched_results[i].end(), std::ostream_iterator<double>(out_results, "\t"));
    //         out_results << std::endl;
    // }

    return 0;


}

int compute3(const imretDataTypes::opt& pOpt, std::vector<std::string>& paths, int topn, float match_thr ){

    std::string rec_path = paths[0];
    std::string GPS_path = paths[1];
    std::string gazes_path = paths[2];
    std::string qr_img_path =  paths[3];

    std::vector<std::vector<double> > matched_results;
    std::vector<std::vector<float> > transformed_labels;
    /////////////////offline parts/////////////////

    imretFuncs offlineParts;
    imretDataTypes::ENV env;
    offlineParts.getEnv(env);

    std::vector<std::string> img_list;
    std::vector<int> assgn, vw2im;


    if(stlplus::folder_empty(env.sim_path )){
        offlineParts.getFileList(img_list);
        offlineParts.simprep_gpu(img_list);
    }

    if(!stlplus::file_exists(pOpt.sadr_vv)){
        offlineParts.create_vv(pOpt);
    }

    if(!stlplus::folder_empty(env.sim_path)){
        img_list = stlplus::folder_wildcard (env.sim_path, "*.sift", false, true);
        std::sort(img_list.begin(), img_list.end(), compareNat);
        offlineParts.simquant(img_list,  pOpt,  assgn, vw2im);

    }


    std::string sim_path;
    offlineParts.get_sim_path(sim_path);


    std::vector<std::string> siftFiles = stlplus::folder_wildcard(sim_path, "*.sift", false, true);

    std::sort(siftFiles.begin(), siftFiles.end(), compareNat);

    imretDataTypes::data_qr data_qr_offline;
    imretDataTypes::data_vv data_vv_offline;
    offlineParts.load_ftrs_to_sim(data_qr_offline.assgn, data_qr_offline.vw2im, data_vv_offline.CN, data_vv_offline.N);

    Eigen::SparseMatrix<float> tfidf;
    offlineParts.tfidf_from_vvQR_fast(data_qr_offline, data_vv_offline, tfidf);


    //  Eigen::SparseMatrix<float> sim;
    //  offlineParts.simget(sim);


    ////////////load labels, gazes and gps////////////////////////////////////////

    std::vector <std::vector<double> > gazes,gps_assgn;
    getGazes(gazes_path,  gazes);
    std::string label_path_db = "../workspace/labels/labels_yy2.txt";
    std::string gps_proc_path = GPS_path;
    std::string gps_assgn_path = stlplus::folder_part(gps_proc_path) +"/"+ stlplus::basename_part(gps_proc_path)+"_assgn.txt";

    if(!stlplus::file_exists(gps_assgn_path)){
        getDBImID(gps_proc_path,  label_path_db,  gps_assgn);
    }else{
        std::ifstream in_gps(gps_assgn_path,std::ofstream::in);
        std::string line;
        while(std::getline(in_gps, line)){
            if(line !="" && line != "\n" && line != "\t"&& line!= " "){
                std::istringstream iss(line);
                std::vector<double> lineElems(7,0);
                int i;
                for(i = 0; i < lineElems.size(); ++i){
                    iss >> lineElems[i];
                }
                gps_assgn.push_back(lineElems);
            }
        }

    }
    ////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////


    //time peroids are measured in this part//


    /////////////////////////////online parts////////////////////////////

    std::string qr_sim_path = qr_img_path + "_sim/" ;
    std::string qr_imfmt = "png";
    //make sure the path ends with separator
    qr_img_path = stlplus::folder_append_separator(paths[3]);

    imretDataTypes::ENV qr_env{qr_img_path,qr_sim_path,qr_imfmt };

    imretFuncs onlineParts(rec_path, qr_env);
    int i,j,k,u;
    std::vector<std::string> qr_img_list,  img_to_proc;

    onlineParts.getFileList(qr_img_list);
    std::sort( qr_img_list.begin(), qr_img_list.end(), compareNat);


    /////////filter the frames with correctly synchronized gazes/////////

    for(i = 0; i < qr_img_list.size(); ++i){
        for(j = 0; j < gazes.size(); ++j){
            if(gazes[j][0] == (double)i){
                img_to_proc.push_back(qr_img_list[i]);
                break;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////

    ///////////check the difference between sift files and img files to decide if there are imgs whose sifts are not extracted//////////
    ///////////useful in real time application ///////////
    //
    // qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, "*.sift", false, true);
    // std::sort( qr_sift_list.begin(), qr_sift_list.end(), compareNat);
    // int diffSiftImg = qr_img_list.size() - qr_sift_list.size();
    // if(diffSiftImg){
    //         img_to_proc.resize(diffSiftImg);
    //         std::copy(qr_img_list.end() - diffSiftImg, qr_img_list.end(),img_to_proc.begin());
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int ii = 0;
    for(ii = 0; ii < img_to_proc.size(); ++ ii){



        if((ii>0) && matched_results[ii-1][3]){
            //////////////////////////algorithm 1: utilize the homography from previous matching///////////////////////////////
            std::cout << "algorithm 1 is used" << std::endl;
            //measure time

            gettimeofday(&starttime, NULL);


            float testx = gazes[ii][1];
            float testy = gazes[ii][2];
            float vertx[4] = {transformed_labels[ii-1][0],transformed_labels[ii-1][1],transformed_labels[ii-1][2],transformed_labels[ii-1][3]};
            float verty[4] = {transformed_labels[ii-1][4],transformed_labels[ii-1][5],transformed_labels[ii-1][6],transformed_labels[ii-1][7]};

            std::vector<double> matched_result_element(4,0);
            matched_result_element[0] = gazes[ii][0];
            matched_result_element[2] = matched_results[ii-1][2];
            if( pnpoly(4, vertx, verty,  testx,testy) ){
                matched_result_element[1] = matched_results[ii-1][1];
                //record if the homography is useful
                matched_result_element[3] = 1;
            }
            matched_results.push_back(matched_result_element);
            gettimeofday(&endtime, NULL);
            double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;


            std::cout << delta << std::endl;


        }

        std::vector<std::string> current_frame;
        current_frame.push_back(img_to_proc[ii]);
        onlineParts.simprep_gpu(current_frame);

        //simquant load sift files
        std::string sift_file_name = "dvw_" + stlplus::basename_part(img_to_proc[ii]) + ".sift";
        std::vector<std::string> qr_sift_list;
        qr_sift_list.push_back(sift_file_name);

        if((ii>0) && matched_results[ii-1][3] && (matched_results.size() == ii)){

            //////////////////////////algorithm 2: use former matched image ///////////////////////////////////////
            /////////////////////////always try algorithm 2 before 3 and abandon it when nothing is matched////////
            std::cout << "algorithm 2 is used" << std::endl;

            gettimeofday(&starttime, NULL);

            MatrixXfr qr_descs_mat;
            flann::Matrix<Scalar> qr_descs_mat_flann;
            onlineParts.load_descs(qr_sim_path+qr_sift_list[0],qr_descs_mat);
            onlineParts.convert2flann(qr_descs_mat, qr_descs_mat_flann);

            std::unique_ptr< flann::Index<Metric> > qr_index;
            qr_index.reset(new flann::Index<Metric> (qr_descs_mat_flann, flann::KDTreeIndexParams(4)));
            qr_index->buildIndex();

            int suggested_img_id =(int) matched_results[ii-1][2];


            openMVG::image::Image<unsigned char> testImage;
            img_list = stlplus::folder_wildcard(env.img_path, "*.png", false, true);
            std::sort(img_list.begin(), img_list.end(), compareNat);
            std::string imagePath = env.img_path + img_list[0];
            openMVG::image::ReadImage( imagePath.c_str(), &testImage);

            MatrixXfr descs_mat;
            flann::Matrix<Scalar> descs_mat_flann;
            offlineParts.load_descs(sim_path + siftFiles[suggested_img_id],descs_mat);
            offlineParts.convert2flann(descs_mat, descs_mat_flann);
            std::unique_ptr< flann::Matrix<int> > indices;
            indices.reset(new flann::Matrix<int>(new int[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
            std::unique_ptr<flann::Matrix<DistanceType> > dists;
            dists.reset(new flann::Matrix<DistanceType>(new float[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
            flann::SearchParams matching_search_params;
            matching_search_params.checks = 256;
            matching_search_params.cores = 0;
            qr_index->knnSearch(descs_mat_flann, *indices, *dists, 2, matching_search_params);



            Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> indices_mat(indices->ptr(), indices->rows,indices->cols);

            MatrixXfr tmp1(descs_mat.rows(),128);
            MatrixXfr tmp2(descs_mat.rows(),128);

            for(k = 0; k < descs_mat.rows(); ++k){
                MatrixXfr tmp(1,128);
                tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,0));
                tmp1.row(k)= tmp.array().square();
                tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,1));
                tmp2.row(k) = tmp.array().square();
            }
            MatrixXfr tmp1_sum;
            MatrixXfr tmp2_sum;
            Eigen::Matrix<bool,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gm;
            tmp1_sum = tmp1.rowwise().sum().eval();
            tmp2_sum = tmp2.rowwise().sum().eval() * match_thr* match_thr;
            gm = tmp1_sum.array() < tmp2_sum.array();

            std::vector<int> gm_ind;

            for(k =0; k<gm.rows(); ++k){
                if(gm(k,0)){
                    gm_ind.push_back(k);
                }
            }

            std::sort(gm_ind.begin(), gm_ind.end());
            std::vector<int> sel;
            std::vector<int> indices_unique;
            indices_unique.push_back(indices_mat(gm_ind[0],0));
            sel.push_back(0);
            for(k = 1; k < gm_ind.size(); ++k){
                if(std::find(indices_unique.begin(), indices_unique.end(),indices_mat(gm_ind[k],0))==indices_unique.end()){
                    indices_unique.push_back(indices_mat(gm_ind[k],0));
                    sel.push_back(k);
                }
            }

            MatrixXfr qr_feats_mat,feats_mat, tc(sel.size(),4), tc_qr(sel.size(),4);
            offlineParts.load_feats(sim_path + siftFiles[suggested_img_id],feats_mat);
            onlineParts.load_feats(qr_sim_path + qr_sift_list[0], qr_feats_mat);

            int n = 0,m = 0;
            for(k =0; k < sel.size(); ++k){
                int idx_tc = gm_ind[sel[k]];
                int idx_tc_qr = indices_mat(gm_ind[k]);
                if(idx_tc<=feats_mat.cols()){
                    tc.row(n) = feats_mat.row(idx_tc);
                    ++n;
                }

                if(idx_tc_qr<=qr_feats_mat.rows()){
                    tc_qr.row(m) = qr_feats_mat.row(indices_mat(gm_ind[sel[k]],1));
                    ++m;
                }
            }

            if(n-1<sel.size()){
                tc.resize(n-1,4);
            }

            if(m-1<sel.size()){
                tc_qr.resize (m-1,4);
            }

            ////////////////////ACRANSAC/////////////////////////////

            std::vector<size_t> vec_inliers;
            typedef openMVG::robust::ACKernelAdaptor<
                    openMVG::homography::kernel::FourPointSolver,
                    openMVG::homography::kernel::AsymmetricError,
                    openMVG::UnnormalizerI,
                    openMVG::Mat3> KernelType;

            openMVG::Mat tc_openMVG;
            openMVG::Mat tc_qr_openMVG;

            if(tc.rows()==tc_qr.rows()){
                tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();

            }else{
                if(tc.rows() < tc_qr.rows()){
                    tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                    tc_qr_openMVG = tc_qr.block(0,0,tc.rows(),2).eval().transpose().cast<double>();
                }else{
                    tc_openMVG = tc.block(0,0,tc_qr.rows(),2).eval().transpose().cast<double>();
                    tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();
                }
            }

            KernelType kernel(
                        tc_openMVG, testImage.Width(), testImage.Height(),
                        tc_qr_openMVG,testImage.Width(), testImage.Height(),
                        false); // configure as point to point error model.

            openMVG::Mat3 H;
            std::pair<double,double> ACRansacOut = ACRANSAC(kernel, vec_inliers, 100, &H,
                                                            std::numeric_limits<double>::infinity(),
                                                            false);







            // std::cout << "qr " << qr_sift_list[i] << std::endl;
            // std::cout <<"db " << siftFiles[qr_simil_ind_sorted[0]] <<" " <<siftFiles[qr_simil_ind_sorted[1]] <<" "<<siftFiles[qr_simil_ind_sorted[2]]<< std::endl;
            // std::cout << "recommended " << siftFiles[matchedImgId] << std::endl;
            // std::cout << std::endl;


            //////////////////////////////////USAC/////////////////////////////////////////
            // HomogEstimator* homog = new HomogEstimator;

            // // initialize the USAC parameters, either from a config file, or from your application
            // homog->initParamsUSAC(cfg);

            // // get input data points/prosac ordering data (if required)
            // // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

            // // set up the estimation problem
            // homog->initDataUSAC(cfg);
            // homog->initProblem(cfg, &point_data);

            // // solve
            // if (!homog->solve()){
            //         return(EXIT_FAILURE);
            // }

            // // do stuff

            // // cleanup
            // homog->cleanupProblem();
            // delete homog;





            /////////////gaze matching//////////////

            std::vector<double> matched_result_element(4,0);
            matched_result_element[0] = gazes[ii][0];
            matched_result_element[2] = suggested_img_id;


            for(j = 0; j < gps_assgn.size(); ++j ){
                if ((double)suggested_img_id == gps_assgn[j][1]){
                    //project frame to gaze with homography

                    openMVG::Mat verts(3,4);
                    verts<< gps_assgn[j][2],gps_assgn[j][2], gps_assgn[j][4],gps_assgn[j][4],
                            gps_assgn[j][5],gps_assgn[j][3],gps_assgn[j][3], gps_assgn[j][5],
                            1,1,1,1;

                    verts = H.inverse()*verts;
                    // openMVG::Mat gazes_mat(3,1);
                    // gazes_mat << gazes[i][1],
                    //         gazes[i][2],
                    //         1;
                    //gazes_mat = bestMatched_H*gazes_mat;

                    float vertx[4] = {(float)(verts(0,0)/verts(2,0)), (float)(verts(0,1)/verts(2,1)),(float)(verts(0,2)/verts(2,2)), (float)(verts(0,3)/verts(2,3))};
                    float verty[4] = {(float)(verts(1,0)/verts(2,0)),  (float)(verts(1,1)/verts(2,1)),(float)(verts(1,2)/verts(2,2)), (float)(verts(1,3)/verts(2,3))};
                    float testx = gazes[ii][1];
                    float testy = gazes[ii][2];
                    //                                 float testx = (float)(gazes_mat(0,0)/gazes_mat(2,0));
                    //                                 float testy = (float)(gazes_mat(1,0)/gazes_mat(2,0));

                    //keep the transformed bounding box in memory
                    std::vector<float> transformed_label_x(std::begin(vertx), std::end(vertx));
                    std::vector<float> transformed_label_y(std::begin(verty), std::end(verty));
                    transformed_label_x.insert(transformed_label_x.end(),transformed_label_y.begin(), transformed_label_y.end());
                    transformed_labels.push_back(transformed_label_x);



                    if( pnpoly(4, vertx, verty,  testx,testy) ){
                        matched_result_element[1] = gps_assgn[j][0];
                        //record if the homography is useful
                        matched_result_element[3] = 1;
                    }
                }
            }

            matched_results.push_back(matched_result_element);



            gettimeofday(&endtime, NULL);
            double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;


            std::cout << delta << std::endl;
            qr_index.reset();
        }

        if((ii==0 ) || (matched_results.size() == ii)){
            //////////////////////////algorithm 3: query the database///////////////////////////////
            std::cout << "algorithm 3 is used" << std::endl;

            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            gettimeofday(&starttime, NULL);


            //                qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, "*.sift", false, true);
            //qr_sift_list = stlplus::folder_wildcard (qr_env.sim_path, sift_file_name, false, true);
            //std::sort(qr_sift_list.begin(), qr_sift_list.end(), compareNat);
            std::vector<int> qr_assgn, qr_vw2im;
            onlineParts.simquant(qr_sift_list, pOpt, qr_assgn, qr_vw2im);


           // gettimeofday(&endtime, NULL);
           // double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;


           // std::cout<< "simquant " <<delta<< std::endl;
            //similarity
            imretDataTypes::data_qr data_qr_online;
            imretDataTypes::data_vv data_vv_online;


            onlineParts.load_ftrs_to_sim(data_qr_online.assgn, data_qr_online.vw2im, data_vv_online.CN, data_vv_online.N);

            // std::copy(data_qr_online.assgn.begin(), data_qr_online.assgn.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << std::endl;


            Eigen::SparseMatrix<float>  tfidf_qr, simil;

            onlineParts.tfidf_from_vvQR_fast(data_qr_online, data_vv_online,tfidf_qr);

            simil = tfidf_qr.transpose()* tfidf;
            //convert simil to dense matrix
            MatrixXfr d_simil(simil);



            //load groundtruth data
            // std::string grnd_truth_fname = rec_path + "groundtruth/image.txt";
            // std::ifstream in_grnd_truth(grnd_truth_fname,std::ios::in);

            // std::vector<int> results;
            // int result;
            // while(in_grnd_truth>>result){
            //         results.push_back(result);
            // }
            // in_grnd_truth.close();

            // double correctVal = 0;
            // double countAll = 118;        //should be changed if the number of query images changes

            //////geometric verification/////


            std::cout << "geometric verification" << std::endl;



            //get image params
            openMVG::image::Image<unsigned char> testImage;
            img_list = stlplus::folder_wildcard(env.img_path, "*.png", false, true);
            std::sort(img_list.begin(), img_list.end(), compareNat);
            std::string imagePath = env.img_path + img_list[0];
            openMVG::image::ReadImage( imagePath.c_str(), &testImage);




            for(i = 0; i < d_simil.rows(); ++i){
                //load the descriptors of the query images
                MatrixXfr qr_descs_mat;
                flann::Matrix<Scalar> qr_descs_mat_flann;
                onlineParts.load_descs(qr_sim_path+qr_sift_list[i],qr_descs_mat);
                onlineParts.convert2flann(qr_descs_mat, qr_descs_mat_flann);

                std::unique_ptr< flann::Index<Metric> > qr_index;
                qr_index.reset(new flann::Index<Metric> (qr_descs_mat_flann, flann::KDTreeIndexParams(4)));
                qr_index->buildIndex();

                std::vector<float> qr_simil ((&(d_simil.row(i).data()[0])), (&(d_simil.row(i).data()[0]) + d_simil.cols()));
                std::vector<size_t> qr_simil_ind_sorted;
                std::vector<float> qr_simil_sorted;
                onlineParts.sort(qr_simil,qr_simil_sorted,qr_simil_ind_sorted);

                std::reverse(std::begin(qr_simil_ind_sorted), std::end(qr_simil_ind_sorted));

                //filter with gps data first by checking the neighbor buidlings of the building suggested by gps data
                std::vector<int> gps_suggested_idx;
                for(k = 0; k < gps_assgn.size(); ++k){
                    if(gps_assgn[k].back()){

                        gps_suggested_idx.push_back((int)gps_assgn[k][1]);
                        if((gps_suggested_idx.size() > 1) && (gps_suggested_idx[gps_suggested_idx.size() - 2] == gps_suggested_idx.back())){
                            gps_suggested_idx.pop_back();
                        }

                    }
                }

                std::vector<int> remain_topn;
                for(j = 0 ; j < topn; ++j){
                    for(k = 0; k < gps_suggested_idx.size(); ++k){
                        if(qr_simil_ind_sorted[j] == gps_suggested_idx[k] ){
                            remain_topn.push_back(qr_simil_ind_sorted[j]);
                        }
                    }
                }


                double dNfa = 999;
                int matchedImgId = -1;

                //                std::cout << gps_suggested_idx.size() << std::endl;
                if(!remain_topn.size()){
                    if(gps_suggested_idx.size() < topn){
                        remain_topn.resize(gps_suggested_idx.size());
                        std::copy (gps_suggested_idx.begin(), gps_suggested_idx.end(), remain_topn.begin());
                    }else{
                        remain_topn.resize(topn);
                        //  std::cout << gps_suggested_idx.size() << std::endl;
                        std::copy (gps_suggested_idx.begin(), gps_suggested_idx.begin() + topn, remain_topn.begin());
                    }
                }

               // std::cout << "remain_topn size "<< remain_topn.size() << std::endl;

                std::vector<openMVG::Mat3> Hs;
                int bestMatchedIdx = 0;
                flann::SearchParams matching_search_params;
                matching_search_params.checks = 256;
                matching_search_params.cores = 0;
                for(j = 0; j < remain_topn.size(); ++j){

                  //  gettimeofday(&starttime,NULL);
                    MatrixXfr descs_mat;
                    // std::cout << siftFiles[remain_topn[j]] << std::endl;
                    flann::Matrix<Scalar> descs_mat_flann;
                    //                std::cout <<i << " " << siftFiles[remain_topn[j]] << std::endl;
                    offlineParts.load_descs(sim_path + siftFiles[remain_topn[j]],descs_mat);
                    offlineParts.convert2flann(descs_mat, descs_mat_flann);
                    std::unique_ptr< flann::Matrix<int> > indices;
                    indices.reset(new flann::Matrix<int>(new int[descs_mat_flann.rows*2], descs_mat_flann.rows,2));
                    std::unique_ptr<flann::Matrix<DistanceType> > dists;
                    dists.reset(new flann::Matrix<DistanceType>(new float[descs_mat_flann.rows*2], descs_mat_flann.rows,2));

                    qr_index->knnSearch(descs_mat_flann, *indices, *dists, 2, matching_search_params);


                    //gettimeofday(&endtime, NULL);
                    //double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;
                    //std::cout<< "matching " << delta << std::endl;
                    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> indices_mat(indices->ptr(), indices->rows,indices->cols);

                    MatrixXfr tmp1(descs_mat.rows(),128);
                    MatrixXfr tmp2(descs_mat.rows(),128);

                    for(k = 0; k < descs_mat.rows(); ++k){
                        MatrixXfr tmp(1,128);
                        tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,0));
                        tmp1.row(k)= tmp.array().square();
                        tmp = descs_mat.row(k) - qr_descs_mat.row(indices_mat(k,1));
                        tmp2.row(k) = tmp.array().square();
                    }
                    MatrixXfr tmp1_sum;
                    MatrixXfr tmp2_sum;
                    Eigen::Matrix<bool,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gm;
                    tmp1_sum = tmp1.rowwise().sum().eval();
                    tmp2_sum = tmp2.rowwise().sum().eval() * match_thr* match_thr;
                    gm = tmp1_sum.array() < tmp2_sum.array();

                    std::vector<int> gm_ind;

                    for(k =0; k<gm.rows(); ++k){
                        if(gm(k,0)){
                            gm_ind.push_back(k);
                        }
                    }

                    std::sort(gm_ind.begin(), gm_ind.end());
                    std::vector<int> sel;
                    std::vector<int> indices_unique;
                    indices_unique.push_back(indices_mat(gm_ind[0],0));
                    sel.push_back(0);
                    for(k = 1; k < gm_ind.size(); ++k){
                        if(std::find(indices_unique.begin(), indices_unique.end(),indices_mat(gm_ind[k],0))==indices_unique.end()){
                            indices_unique.push_back(indices_mat(gm_ind[k],0));
                            sel.push_back(k);
                        }
                    }

                    MatrixXfr qr_feats_mat,feats_mat, tc(sel.size(),4), tc_qr(sel.size(),4);
                    offlineParts.load_feats(sim_path + siftFiles[remain_topn[j]],feats_mat);
                    onlineParts.load_feats(qr_sim_path + qr_sift_list[i], qr_feats_mat);

                    int n = 0,m = 0;
                    for(k =0; k < sel.size(); ++k){
                        int idx_tc = gm_ind[sel[k]];
                        int idx_tc_qr = indices_mat(gm_ind[k]);
                        if(idx_tc<=feats_mat.cols()){
                            tc.row(n) = feats_mat.row(idx_tc);
                            ++n;
                        }

                        if(idx_tc_qr<=qr_feats_mat.rows()){
                            tc_qr.row(m) = qr_feats_mat.row(indices_mat(gm_ind[sel[k]],1));
                            ++m;
                        }
                    }

                    if(n-1<sel.size()){
                        tc.resize(n-1,4);
                    }

                    if(m-1<sel.size()){
                        tc_qr.resize (m-1,4);
                    }




                    ////////////////////ACRANSAC/////////////////////////////

                    std::vector<size_t> vec_inliers;
                    typedef openMVG::robust::ACKernelAdaptor<
                            openMVG::homography::kernel::FourPointSolver,
                            openMVG::homography::kernel::AsymmetricError,
                            openMVG::UnnormalizerI,
                            openMVG::Mat3> KernelType;

                    openMVG::Mat tc_openMVG;
                    openMVG::Mat tc_qr_openMVG;

                    if(tc.rows()==tc_qr.rows()){
                        tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                        tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();

                    }else{
                        if(tc.rows() < tc_qr.rows()){
                            tc_openMVG  = tc.block(0,0,tc.rows(), 2).eval().transpose().cast<double>();
                            tc_qr_openMVG = tc_qr.block(0,0,tc.rows(),2).eval().transpose().cast<double>();
                        }else{
                            tc_openMVG = tc.block(0,0,tc_qr.rows(),2).eval().transpose().cast<double>();
                            tc_qr_openMVG = tc_qr.block(0,0, tc_qr.rows(), 2).eval().transpose().cast<double>();
                        }
                    }

                    KernelType kernel(
                                tc_openMVG, testImage.Width(), testImage.Height(),
                                tc_qr_openMVG,testImage.Width(), testImage.Height(),
                                false); // configure as point to point error model.

                    openMVG::Mat3 H;
                    std::pair<double,double> ACRansacOut = ACRANSAC(kernel, vec_inliers, 100, &H,
                                                                    std::numeric_limits<double>::infinity(),
                                                                    false);

                    Hs.push_back(H);

                    if(ACRansacOut.first <= dNfa){
                        dNfa = ACRansacOut.first;
                        matchedImgId = remain_topn[j];
                        bestMatchedIdx = j;
                    }


                    //gettimeofday(&starttime, NULL);


                    //delta = ((starttime.tv_sec  - endtime.tv_sec) * 1000000u -  endtime.tv_usec + starttime.tv_usec) / 1.e6;
                    //std::cout << "acransac "<< delta<< std::endl;

                    // std::cout << "qr " << qr_sift_list[i] << std::endl;
                    // std::cout <<"db " << siftFiles[qr_simil_ind_sorted[0]] <<" " <<siftFiles[qr_simil_ind_sorted[1]] <<" "<<siftFiles[qr_simil_ind_sorted[2]]<< std::endl;
                    // std::cout << "recommended " << siftFiles[matchedImgId] << std::endl;
                    // std::cout << std::endl;


                    //////////////////////////////////USAC/////////////////////////////////////////
                    // HomogEstimator* homog = new HomogEstimator;

                    // // initialize the USAC parameters, either from a config file, or from your application
                    // homog->initParamsUSAC(cfg);

                    // // get input data points/prosac ordering data (if required)
                    // // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

                    // // set up the estimation problem
                    // homog->initDataUSAC(cfg);
                    // homog->initProblem(cfg, &point_data);

                    // // solve
                    // if (!homog->solve()){
                    //         return(EXIT_FAILURE);
                    // }

                    // // do stuff

                    // // cleanup
                    // homog->cleanupProblem();
                    // delete homog;


                }


                openMVG::Mat3 bestMatched_H = Hs[bestMatchedIdx];

                /////////////gaze matching//////////////

                std::vector<double> matched_result_element(4,0);
                matched_result_element[0] = gazes[ii][0];
                matched_result_element[2] = matchedImgId;


                for(j = 0; j < gps_assgn.size(); ++j ){
                    if ((double)matchedImgId == gps_assgn[j][1]){
                        //project frame to gaze with homography

                        openMVG::Mat verts(3,4);
                        verts<< gps_assgn[j][2],gps_assgn[j][2], gps_assgn[j][4],gps_assgn[j][4],
                                gps_assgn[j][5],gps_assgn[j][3],gps_assgn[j][3], gps_assgn[j][5],
                                1,1,1,1;

                        verts = bestMatched_H.inverse()*verts;
                        // openMVG::Mat gazes_mat(3,1);
                        // gazes_mat << gazes[i][1],
                        //         gazes[i][2],
                        //         1;
                        //gazes_mat = bestMatched_H*gazes_mat;

                        float vertx[4] = {(float)(verts(0,0)/verts(2,0)), (float)(verts(0,1)/verts(2,1)),(float)(verts(0,2)/verts(2,2)), (float)(verts(0,3)/verts(2,3))};
                        float verty[4] = {(float)(verts(1,0)/verts(2,0)),  (float)(verts(1,1)/verts(2,1)),(float)(verts(1,2)/verts(2,2)), (float)(verts(1,3)/verts(2,3))};
                        float testx = gazes[ii][1];
                        float testy = gazes[ii][2];
                        //                                 float testx = (float)(gazes_mat(0,0)/gazes_mat(2,0));
                        //                                 float testy = (float)(gazes_mat(1,0)/gazes_mat(2,0));

                        //keep the transformed bounding box in memory
                        std::vector<float> transformed_label_x(std::begin(vertx), std::end(vertx));
                        std::vector<float> transformed_label_y(std::begin(verty), std::end(verty));
                        transformed_label_x.insert(transformed_label_x.end(),transformed_label_y.begin(), transformed_label_y.end());
                        transformed_labels.push_back(transformed_label_x);

                        // std::cout << "bounding box:" << std::endl;
                        // std::cout << verts << std::endl;
                        // std::cout << "gaze:" << std::endl;
                        // std::cout << testx << " " << testy << std::endl;

                        if( pnpoly(4, vertx, verty,  testx,testy) ){
                            matched_result_element[1] = gps_assgn[j][0];
                            //record if the homography is useful
                            matched_result_element[3] = 1;
                        }
                    }
                }
                matched_results.push_back(matched_result_element);
                // std::cout << siftFiles[matchedImgId] << std::endl;
                // for( j = 0; j < results.size(); ++j){
                //         if(matchedImgId == j){
                //                 if(results[j] == 1){
                //                        correctVal++;
                //                        break;
                //                 }
                //                        //countAll++;
                //         }

                // }

                qr_index.reset();


            } //to loop through the similarity matrix row by row, in this case, there is only one row because there is only one frame




            gettimeofday(&endtime, NULL);
            double delta = ((endtime.tv_sec  - starttime.tv_sec) * 1000000u +  endtime.tv_usec - starttime.tv_usec) / 1.e6;
            std::cout << delta << std::endl;

        }//algorithm 3

        //delete ftrs_to_sim file
        std::string ftrs_to_sim_path = qr_sim_path + "ftrs_to_sim";
        stlplus::file_delete (ftrs_to_sim_path);


    }// to feature-matching one frame by another



    //write out the final results
    // std::cout << "writing out final results" << std::endl;
    // std::string final_results_path = "../workspace/results/results.txt";
    // std::ofstream out_results (final_results_path, std::ofstream::out);
    // for (size_t i=0; i < matched_results.size(); ++i )  {
    //         std::copy( matched_results[i].begin(), matched_results[i].end(), std::ostream_iterator<double>(out_results, "\t"));
    //         out_results << std::endl;
    // }

    return 0;
}

//////////////////////////////////////////////////////////////////////////test printout////////////////////////////////////////////////////////////////////////////////

int  printout(){

    return 0;
}


#endif
















