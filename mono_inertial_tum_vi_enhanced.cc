/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>
#include <iomanip>      // 用于格式化输出
#include <signal.h>     // 信号处理
#include <cmath>

#include<opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/features2d.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

extern float g_current_quality;

// 全局变量（用于信号处理和传递质量/机动信息）
FILE* g_stats_fp = nullptr;
ORB_SLAM3::System* g_pSLAM = nullptr;
volatile sig_atomic_t g_shutdown_requested = 0;

// 用于向后端优化传递当前帧的质量和机动性（后续在优化代码中读取）
//float g_current_quality = 1.0f;
float g_current_motion = 0.0f;   // 机动性指标（例如角速度模长）

void sigint_handler(int sig) {
    g_shutdown_requested = 1;
}

void LoadImagesTUMVI(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

cv::Mat GammaEnhance(const cv::Mat& img, double gamma)
{
    cv::Mat lut(1,256,CV_8U);

    for(int i=0;i<256;i++)
        lut.at<uchar>(i) =
            pow(i/255.0, gamma)*255.0;

    cv::Mat result;
    cv::LUT(img,lut,result);

    return result;
}


double ttrack_tot = 0;

int main(int argc, char **argv)
{
    // 注册信号处理
    signal(SIGINT, sigint_handler);

    const int num_seq = (argc-3)/3;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName = ((argc % 3) == 1);

    string file_name;
    if (bFileName)
        file_name = string(argv[argc-1]);

    cout << "file name: " << file_name << endl;

    if(argc < 6)
    {
        cerr << endl << "Usage: ./mono_inertial_tum_vi_enhanced path_to_vocabulary path_to_settings path_to_image_folder_1 path_to_times_file_1 path_to_imu_data_1 (path_to_image_folder_2 path_to_times_file_2 path_to_imu_data_2 ... path_to_image_folder_N path_to_times_file_N path_to_imu_data_N) (trajectory_file_name)" << endl;
        return 1;
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImagesTUMVI(string(argv[3*(seq+1)]), string(argv[3*(seq+1)+1]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(string(argv[3*(seq+1)+2]), vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first
        while(vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true, 0, file_name);
    g_pSLAM = &SLAM;

    // 打开统计文件（重写模式）
    string stats_path = "/home/lee/ORB_SLAM3/save_result/mono-inertial/TUM-VI/gww/gww5_mono-inertial.txt";
    // 确保目录存在，否则创建
    // 这里简单处理，如果 fopen 失败则输出错误
    g_stats_fp = fopen(stats_path.c_str(), "w");
    if (!g_stats_fp) {
        cerr << "Failed to open stats file: " << stats_path << endl;
    }

    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    int proccIm = 0;
    // 预先创建 ORB 提取器（用于统计特征点数量）
    cv::Ptr<cv::ORB> orb_extractor = cv::ORB::create(1000, 1.2f, 8);

    for (seq = 0; seq<num_seq; seq++)
    {
        // Main loop
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;

        int low_quality_count = 0;
        

        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            if (g_shutdown_requested) break;

            // 读取图像（灰度图）
            im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_GRAYSCALE);
            double tframe = vTimestampsCam[seq][ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: " << vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            // ---------- 自适应图像预处理模块（第一创新点）----------
            int64_t start_ticks = cv::getTickCount();

            // 图像已是灰度图，直接使用
            cv::Mat gray = im;

            // 统计原始图像特征点数量
            std::vector<cv::KeyPoint> keypoints;
            orb_extractor->detect(gray, keypoints);
            //cv::FAST(gray, keypoints, 30, true);
            int nfeatures_orig = keypoints.size();

            // 图像质量评估
            // 计算平均梯度
            cv::Mat grad_x, grad_y;
            cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
            cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

            cv::Mat grad_mag;
            cv::magnitude(grad_x, grad_y, grad_mag);

            double grad_mean = cv::mean(grad_mag)[0];

            // 归一化参数
            double F = std::min(nfeatures_orig / 1000.0 , 1.0);
            double G = std::min(grad_mean / 30.0 , 1.0);

            double quality = 0.6 * F + 0.4 * G;
            double thresholdVal = 0.4; // 阈值可调

            // 保存当前帧质量到全局变量（供优化使用）
            g_current_quality = quality;

            cv::Mat img_to_track;
            int nfeatures_enhanced = 0;
            bool enhanced = false;
            /*
            if(quality < thresholdVal)
                low_quality_count++;
            else
                low_quality_count = 0;
            if (low_quality_count >= 3 && F < 0.35)
            */ 
            if (quality < thresholdVal)
            {
                enhanced = true;
                
                double gamma = 1.0 - 2*(0.5-quality);
                gamma = std::max(0.4, std::min(gamma,0.8));
                cv::Mat enhanced_gray = GammaEnhance(gray, gamma);
                /*
                //原处理全局CLAHE增强
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                cv::Mat enhanced_gray;
                clahe->apply(gray, enhanced_gray);
                */
                // 统计增强后的特征点数量
                std::vector<cv::KeyPoint> keypoints_enh;
                orb_extractor->detect(enhanced_gray, keypoints_enh);
                //cv::FAST(enhanced_gray, keypoints_enh, 30, true);
                nfeatures_enhanced = keypoints_enh.size();



                img_to_track = enhanced_gray;
            } 
            else 
            {
                enhanced = false;
                nfeatures_enhanced = nfeatures_orig;
                img_to_track = gray;
            }

            // 计算预处理耗时
            int64_t end_ticks = cv::getTickCount();
            double time_ms = (end_ticks - start_ticks) / cv::getTickFrequency() * 1000.0;

            // 打印信息（可选）
            cout << fixed << setprecision(3);
            cout << "Preprocess: quality=" << quality
                 << ", count=" << low_quality_count
                 << ", enhanced=" << enhanced
                 << ", orig_features=" << nfeatures_orig
                 << ", enh_features=" << nfeatures_enhanced
                 << ", time=" << fixed << setprecision(2) << time_ms << " ms" << endl;

            // 写入统计文件
            if (g_stats_fp) {
                fprintf(g_stats_fp, "%.3f %d %d %d %.2f\n",
                        quality, enhanced, nfeatures_orig, nfeatures_enhanced, time_ms);
                fflush(g_stats_fp);
            }
            // ---------- 预处理模块结束 ----------

            // 加载 IMU 数据（原始代码已有）
            vImuMeas.clear();
            if(ni>0)
            {
                while(vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,
                                                             vAcc[seq][first_imu[seq]].y,
                                                             vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,
                                                             vGyro[seq][first_imu[seq]].y,
                                                             vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
            }

            // 计算机动性指标（例如当前角速度模长），可用于调整 IMU 权重
            // 取最近一个 IMU 数据的角速度模长作为机动性代表
            float motion = 0.0f;
            if (!vImuMeas.empty()) {
                const auto& last_imu = vImuMeas.back();
                float wx = last_imu.w.x(), wy = last_imu.w.y(), wz = last_imu.w.z();
                motion = sqrt(wx*wx + wy*wy + wz*wz);
            }
            g_current_motion = motion;  // 保存到全局变量，供优化使用

            // 可选：图像缩放（原始代码已有）
            if(imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
                // ... 省略缩放计时代码（原样保留）
#endif
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(img_to_track, img_to_track, cv::Size(width, height));
            }

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // 将图像和 IMU 数据传递给 SLAM 系统
            SLAM.TrackMonocular(img_to_track, tframe, vImuMeas);

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            vTimesTrack[ni] = ttrack;

            // 等待以匹配帧率
            double T=0;
            if(ni < nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1] - tframe;
            else if(ni > 0)
                T = tframe - vTimestampsCam[seq][ni-1];

            if(ttrack < T)
                usleep((T-ttrack)*1e6);
        }

        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }
    }

    // 关闭统计文件
    if (g_stats_fp) {
        fclose(g_stats_fp);
        g_stats_fp = nullptr;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages[0]; ni++)
        totaltime += vTimesTrack[ni];
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl;
    cout << "mean tracking time: " << totaltime/proccIm << endl;

    return 0;
}

void LoadImagesTUMVI(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    cout << strImagePath << endl;
    cout << strPathTimes << endl;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);

        if(!s.empty())
        {
            if (s[0] == '#')
                continue;

            int pos = s.find(' ');
            string item = s.substr(0, pos);

            vstrImages.push_back(strImagePath + "/" + item + ".png");
            double t = stod(item);
            vTimeStamps.push_back(t/1e9);
        }
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
}