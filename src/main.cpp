#include "mimo_unet.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>

// 使用C++17标准的文件系统库，简化文件和路径操作
namespace fs = std::filesystem;

int main() {
    // 创建TensorRT日志对象
    Logger logger;

    // 定义模型engine文件、输入图片目录、输出结果目录的路径
    const std::string enginePath = "/home/axi/deblur_result/MIMO-Unet4+8/Final_trt.engine";
    const std::string inputDir   = "/home/axi/deblur_result/MIMO-Unet4+8/model_test_result/blur";
    const std::string outputDir  = "/home/axi/deblur_result/MIMO-Unet4+8/engine_infer_result";

    // 如果输出目录不存在则自动创建
    fs::create_directories(outputDir);

    // 指定网络输入张量的形状（NCHW格式：batch, channel, height, width）
    nvinfer1::Dims4 shape{1, 1, 2000, 2000};

    // 创建MimoUnet推理对象，并加载engine模型
    MimoUnet mimo(enginePath, shape, logger);

    // 收集所有待处理图片的路径（只收集png/jpg/jpeg格式）
    std::vector<fs::path> imgPaths;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) continue; // 跳过非文件项
        auto ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") imgPaths.push_back(entry.path());
    }
    
	// 如果没有找到任何图片，则报错并退出程序
	if (imgPaths.empty()) {
        std::cerr << "未找到任何图片！" << std::endl;
        return -1;
    }
	
	std::cout << "共找到 " << imgPaths.size() << " 张图片。" << std::endl;

	// ----------- 推理预热阶段（可多次）-----------
	// 加载第一张图片用于预热，提高后续推理速度和稳定性
	cv::Mat warm_img = cv::imread(imgPaths[0].string(), cv::IMREAD_GRAYSCALE);
	warm_img.convertTo(warm_img, CV_32F, 1.0 / 255.0);      // 转为float类型并归一化到[0,1]
	mimo.infer(warm_img);                                   // 执行一次推理

	double total_time_ms = 0.;                              // 总耗时统计变量
	int count = imgPaths.size();                            // 图片总数

	// ----------- 正式批量推理循环 -----------
	for (const auto& imgPath : imgPaths) {
	    cv::Mat img = cv::imread(imgPath.string(), cv::IMREAD_GRAYSCALE);
	    if (img.empty()) {
	        std::cerr << "读取失败: " << imgPath << std::endl; 
	        continue;
	    }
	    img.convertTo(img, CV_32F, 1.0 / 255.0);            // 转为float类型并归一化

	    cudaEvent_t start, stop;                             // CUDA事件用于计时
	    cudaEventCreate(&start); 
		cudaEventCreate(&stop);
	    cudaEventRecord(start);

	    float* out = mimo.infer(img);                        // 调用MimoUnet进行推理

	    cudaEventRecord(stop); 
		cudaEventSynchronize(stop);
	    float ms = 0.; 
	    cudaEventElapsedTime(&ms, start, stop);              // 获取本次推理耗时(ms)
	    total_time_ms += ms;
	    cudaEventDestroy(start); 
		cudaEventDestroy(stop);
		
	    cv::Mat outMat(mimo.inputHeight(), mimo.inputWidth(), CV_32F, out);   // 输出数据转为OpenCV矩阵形式
	    cv::Mat out8; 
		outMat.convertTo(out8, CV_8U, 255.0);                // 恢复到uint8灰度图像

	    fs::path save_path = fs::path(outputDir) / imgPath.filename();         // 构造保存路径
	    cv::imwrite(save_path.string(), out8);                                 // 保存去模糊后的结果图像

	    std::cout << "已处理:" << imgPath.filename() 
	              << " 耗时:"   << ms 
				  << " ms"     << std :: endl ;
	}

	std :: cout <<"全部完成! 总共处理 "<<count<<" 张图片，平均单张推理耗时:"
	            <<(total_time_ms/count)<<" ms"<<std :: endl ;

	return 0 ;
}