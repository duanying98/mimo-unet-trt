#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unordered_map>

// 日志类，用于TensorRT内部信息输出（只打印非INFO级别日志）
class Logger : public nvinfer1::ILogger {
public:
    // 重写log方法，severity为日志等级，msg为日志内容
    void log(Severity severity, const char* msg) noexcept override;
};

// MimoUnet类：封装MIMO-Unet TensorRT推理流程
class MimoUnet {
public:
    /**
     * @brief 构造函数，加载engine并初始化推理环境
     * @param enginePath   TensorRT序列化模型文件路径
     * @param inputShape   输入张量形状（Dims4: NCHW）
     * @param logger       日志对象引用
     */
    MimoUnet(const std::string& enginePath, const nvinfer1::Dims4& inputShape, Logger& logger);

    /**
     * @brief 析构函数，释放所有资源（显存、上下文等）
     */
    ~MimoUnet();

    /**
     * @brief 对单张图片进行推理
     * @param img          输入图片（需为CV_32F类型，与网络输入一致）
     * @param outHostBuf   可选参数：外部输出缓冲区指针（如不传则内部分配并返回指针）
     * @return             指向输出数据的float指针（host端内存）
     */
    float* infer(const cv::Mat& img, float* outHostBuf = nullptr);

    /**
     * @brief 获取网络输入宽度
     */
    int inputWidth() const;

    /**
     * @brief 获取网络输入高度
     */
    int inputHeight() const;

private:
    /**
     * @brief 从磁盘加载engine文件到内存buffer
     */
    std::vector<char> loadEngine(const std::string& path);

    Logger& mLogger;                        // 日志对象引用
    nvinfer1::Dims4 mInputShape;            // 输入张量形状

    nvinfer1::IRuntime* mRuntime{nullptr};              // TensorRT运行时对象指针
    nvinfer1::ICudaEngine* mEngine{nullptr};            // TensorRT引擎对象指针
    nvinfer1::IExecutionContext* mContext{nullptr};     // 推理上下文对象指针

    std::vector<void*> mBuffers;             // 存放所有输入/输出GPU buffer的指针数组
    std::vector<int> mSizes;                 // 每个buffer对应的数据元素数量

    cudaStream_t mStream{};                  // CUDA流，用于异步操作

    std::vector<float> mHostOut;             // host端输出缓存区

	// 输入/输出tensor名称与其在buffers中的索引映射表，以及实际用到的名字字符串 
	// （适配多输入多输出情况，一般只用第一个input/output即可）
	std::unordered_map<std::string, int> mInputNameIdx, mOutputNameIdx;
	std::string mInputName, mOutputName;

public:
	/**
	 * @brief 获取当前主输入tensor在buffers中的索引号 
	 */
	int getInputIndex() const;

	/**
	 * @brief 获取当前主输出tensor在buffers中的索引号 
	 */
	int getOutputIndex() const;

    /**
     * @brief 设置主输出tensor的名字（用于多输出模型时切换）
     * @param outputName 主输出tensor的名称
     */
    void setOutputName(const std::string& outputName);

    /**
     * @brief 设置主输入tensor的名字（用于多输入模型时切换）
     * @param inputName 主输入tensor的名称
     */
    void setInputName(const std::string& inputName);
};