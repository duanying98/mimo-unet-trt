#include "mimo_unet.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

using namespace nvinfer1;

// Logger实现：只打印非INFO级别日志
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
}

// MimoUnet构造函数：加载engine、分配buffer、初始化推理环境
MimoUnet::MimoUnet(const std::string& enginePath, const Dims4& inputShape, Logger& logger)
    : mLogger(logger), mInputShape(inputShape)
{
    // 1. 加载序列化engine文件到内存
    auto engineData = loadEngine(enginePath);

    // 2. 创建TensorRT运行时对象
    mRuntime = createInferRuntime(mLogger);

    // 3. 反序列化engine数据为ICudaEngine对象
    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), engineData.size());

    // 4. 创建推理上下文（IExecutionContext）
    mContext = mEngine->createExecutionContext();

    // 5. 设置输入张量形状（适用于动态shape模型）
    mContext->setInputShape("input", inputShape);

    // 6. 分配所有输入/输出buffer，并记录其大小与索引映射关系
    int nb = mEngine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = mEngine->getIOTensorName(i);      // 获取tensor名称
        Dims dims = mContext->getTensorShape(name);          // 获取tensor实际shape

        int vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];   // 元素总数=各维度乘积

        size_t bytes = vol * sizeof(float);
        void* ptr;
        cudaMalloc(&ptr, bytes);                             // 在GPU上分配buffer空间

        mBuffers.push_back(ptr);
        mSizes.push_back(vol);

        std::string sname(name);
        if (mEngine->getTensorIOMode(name) == TensorIOMode::kINPUT)
            mInputNameIdx[sname] = i;                        // 输入名与索引映射表
        else
            mOutputNameIdx[sname] = i;                       // 输出名与索引映射表
    }
    
	// 创建CUDA流，用于异步拷贝和推理操作，提高效率
	cudaStreamCreate(&mStream);

	// 默认选用第一个输入/输出名字（适合单输入单输出场景）
	if (!mInputNameIdx.empty())   mInputName  = mInputNameIdx.begin()->first;
	if (!mOutputNameIdx.empty())  mOutputName = mOutputNameIdx.begin()->first;
}

// 析构函数：释放所有资源，包括GPU buffer、CUDA流、TRT对象等，防止内存泄漏
MimoUnet::~MimoUnet() {
	for (void* b : mBuffers) cudaFree(b);         // 释放所有GPU buffer显存
	cudaStreamDestroy(mStream);                   // 销毁CUDA流
	delete mContext; delete mEngine; delete mRuntime;   // 删除TRT相关对象指针（注意顺序）
}

/**
 * @brief 推理接口，对单张图片进行前向计算并返回host端输出指针。
 * @param img         输入图片（需为CV_32F类型，与网络输入一致）
 * @param outHostBuf  可选参数：外部输出缓冲区指针，如不传则内部自动分配并返回指针。
 */
float* MimoUnet::infer(const cv::Mat& img, float* outHostBuf) {
	//该模型输入只有一个，不设置

	//该模型输出有三个，设置主输出为output2
	setOutputName("output2");
	
	int inIdx  = getInputIndex();     // 获取主输入buffer索引号
	int outIdx = getOutputIndex();    // 获取主输出buffer索引号

	// 将图片数据从host拷贝到device buffer（异步方式）
	cudaMemcpyAsync(mBuffers[inIdx], img.ptr<float>(),
					sizeof(float)*mSizes[inIdx], cudaMemcpyHostToDevice, mStream);

	// 调用TensorRT执行推理（enqueueV2支持异步流）
	bool ok=mContext->enqueueV2(mBuffers.data(),mStream,nullptr); 
	if(!ok){
	    std::cerr<<"推理失败！"<<std::endl;
	}

	// 如果没有外部输出缓存，则使用内部vector作为host端缓存区，并返回其指针。
	if(outHostBuf==nullptr){
	   	mHostOut.resize(mSizes[outIdx]);
	   	outHostBuf=mHostOut.data();
	}
	
	// 将device上的结果拷贝回host端缓存区，同样采用异步方式，然后同步等待完成。
	cudaMemcpyAsync(outHostBuf,mBuffers[outIdx],
	                sizeof(float)*mSizes[outIdx],cudaMemcpyDeviceToHost,mStream);
	cudaStreamSynchronize(mStream);

	return outHostBuf;
}

// 返回网络输入宽度（W），即NCHW中的W值。方便后续reshape或保存图片等操作。
int MimoUnet::inputWidth()const { return	mInputShape.d[3]; }

// 返回网络输入高度（H），即NCHW中的H值。
int MimoUnet::inputHeight()const { return	mInputShape.d[2]; }

/**
 * @brief 从磁盘读取engine文件内容到内存vector<char>中，用于反序列化。
 */
std :: vector<char> MimoUnet :: loadEngine(const std :: string & path){
	std :: ifstream file(path,std :: ios :: binary|std :: ios :: ate); 
	size_t size=file.tellg(); file.seekg(0,file.beg); 
	std :: vector<char> buffer(size); file.read(buffer.data(),size); return buffer ;
}

/**
 * @brief 获取主输入tensor在buffers数组中的索引号。一般只用第一个input即可。
 */
int MimoUnet :: getInputIndex()const { 
	return !mInputName.empty()?  	mInputNameIdx.at(mInputName):0;
}

/**
 * @brief 获取主输出tensor在buffers数组中的索引号。一般只用第一个output即可。
 */
int MimoUnet :: getOutputIndex()const { 
	return !mOutputName.empty()?  	mOutputNameIdx.at(mOutputName):1;
}

/**
 * @brief 设置主输出tensor的名字（用于多输出模型时切换）
 */
void MimoUnet::setOutputName(const std::string& outputName) {
    if (mOutputNameIdx.find(outputName) == mOutputNameIdx.end()) {
        throw std::runtime_error("输出名 " + outputName + " 不存在！");
    }
    mOutputName = outputName;
}
/**
 * @brief 设置主输入tensor的名字（用于多输入模型时切换）
 */
void MimoUnet::setInputName(const std::string& inputName) {
    if (mInputNameIdx.find(inputName) == mInputNameIdx.end()) {
        throw std::runtime_error("输入名 " + inputName + " 不存在！");
    }
    mInputName = inputName;
}