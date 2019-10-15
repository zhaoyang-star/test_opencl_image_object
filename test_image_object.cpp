#include <iostream>
#include <cstring>
#include <iterator>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include "CL/cl2.hpp"
#define BIN_PATH "./image_object_kernel.bin" /* bin file created by kernel source code */
#define KERNEL_SOURCE "./image_object_kernel.cl" /* kernel source code file */
static std::once_flag create_opencl_stuff_once;

#include <fstream>
/* convert the kernel file into a string */
static int convertToString(const char* filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return -1;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout << "Error: failed to open file: " << filename << std::endl;
    return -1;
}

static int commonReadFile(const char* filename, char** buffer, size_t& length)
{
    size_t size;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        length = size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        *buffer = new char[size];
        if (!*buffer)
        {
            f.close();
            return -1;
        }
        f.read(*buffer, fileSize);
        f.close();
        return 0;
    }
    std::cout << "Error: failed to open file: " << filename << std::endl;
    return -1;
}

int main()
{
    for (int iter = 0; iter < 5; iter++)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        thread_local static cl::CommandQueue mQueue;
        thread_local static cl::Kernel mTestImage;
        thread_local static cl::Image2D mInput;
        thread_local static cl::Image2D mOutput;

        thread_local static float* input;
        thread_local static float* output;

        int mN = 2;
        int mH = 3;
        int mW = 4;
        int mC = 8;

        std::call_once(create_opencl_stuff_once, [&]() -> bool
        {
            // 1. get all platforms (drivers)
            std::vector<cl::Platform> allPlatforms;
            cl::Platform::get(&allPlatforms);
            if (allPlatforms.size() == 0)
            {
                std::cerr << "No platforms found. Check OpenCL installation!" << std::endl;
                return false;
            }
            cl::Platform defaultPlatform = allPlatforms[0];
            std::cout << "Using platform: "
                    << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::cout << "Platform version: "
                    << defaultPlatform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

            // 2. get target device(GPU) of the default platform
            std::vector<cl::Device> allDevices;
            defaultPlatform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
            if (allDevices.size() == 0)
            {
                std::cerr << "No GPU devices found. Check OpenCL installation!" << std::endl;
                return false;
            }
            cl::Device targetDevice = allDevices[0];
            std::cout << "Using device: " << targetDevice.getInfo<CL_DEVICE_NAME>() << "\n"
                    << "SVM capabilities: " << targetDevice.getInfo<CL_DEVICE_SVM_CAPABILITIES>()
                    << std::endl;

            // 3. create context
            cl::Context context({targetDevice});

            // 4. 5. create & build the program
            size_t binarySize;
            cl::Program::Binaries binary;
            bool useCache = false; // FIXME:
            if (useCache)
            {
                /* read binary created by kernel source code */
                char* buffer;
                int ret = commonReadFile(BIN_PATH, &buffer, binarySize);
                if ( ret != 0)
                {
                    std::cerr << "Error open kernel binary file!" << std::endl;
                    return false;
                }
                binary.resize(1);
                binary.at(0).resize(binarySize);
                memcpy(binary.at(0).data(), buffer, sizeof(char) * binarySize);
            }
            else
            {
                /* read kernel source code to string */
                cl::Program::Sources sources;
                std::string kernelCode;
                if (convertToString(KERNEL_SOURCE, kernelCode) != 0)
                {
                    std::cerr << "Error convert OpenCL kernel code to string!" << std::endl;
                    return false;
                }
                sources.push_back({kernelCode.c_str(), kernelCode.length()});

                /* create program from source and build it */
                cl::Program program(context, sources);
                if (program.build({targetDevice}) != CL_SUCCESS)
                {
                    std::cerr << "Error building: "
                            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(targetDevice) << std::endl;
                    return false;
                }
                /* get kernel binary and store it to disk */
                program.getInfo(CL_PROGRAM_BINARIES, &binary);
                binarySize = program.getInfo<CL_PROGRAM_BINARY_SIZES>().at(0);
                FILE* f = fopen(BIN_PATH, "w");
                fwrite(binary.at(0).data(), binarySize, 1, f);
                fclose(f);
                std::cout << "kernel code has been stored as "
                            "a binary file successfully ^_^" << std::endl;
            }
            /* create program from binary and build it */
            const std::vector<cl::Device> device = {targetDevice};
            cl::Program program(context, device, binary);
            if (program.build(device) != CL_SUCCESS)
            {
                std::cerr << "Error building: "
                        << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(targetDevice) << std::endl;
                return false;
            }

            // 6. create buffers on device (allocate space on GPU)
            cl::ImageFormat format(CL_RGBA, CL_FLOAT);
            cl::Image2D imInput(context, CL_MEM_READ_ONLY, format, mW * ((mC + 3) / 4), mN * mH);
            mInput = imInput;
            cl::Image2D imOutput(context, CL_MEM_WRITE_ONLY, format, mW * ((mC + 3) / 4), mN * mH);
            mOutput = imOutput;

            input = new float[mN*mH*mW*mC];
            output = new float[mN*mH*mW*mC];
            int i = 0;
            for (int n = 0; n < mN; n++)
            {
                for (int h = 0; h < mH; h++)
                {
                    for (int w = 0; w < mW; w++)
                    {
                        for (int c = 0; c < mC; c++)
                        {
                            input[i] = i * 1.f;
                            output[i] = i * 1.f;
			    std::cout << input[i] << "\t";
                            i++;
                        }
                    }
		    std::cout << std::endl;
                }
            }
	    std::cout << "-------------------" << std::endl;

            // 7. create a queue (a queue of commands that the GPU will execute)
            cl::CommandQueue queue(context, targetDevice, cl::QueueProperties::Profiling);
            mQueue = queue;
            mTestImage = cl::Kernel(program, "image_object_kernel");
            size_t maxWorkGroupSize;
            mTestImage.getWorkGroupInfo(targetDevice, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize);
            std::cout << "maxWorkGroupSize: " << maxWorkGroupSize << std::endl;
            std::cout << "OpenCL stuff create done!" << std::endl;
            return true;
        });
        auto t_end1 = std::chrono::high_resolution_clock::now();
        std::cout << "Wall time of OpenCL create: "
                << std::chrono::duration<double, std::milli>(t_end1 - t_start).count() << " ms" << std::endl;

        // 8. push data from CPU to GPU
        const std::array<cl::size_type, 3> origin = {0};
        const std::array<cl::size_type, 3> region_input = {(cl::size_type)(mW * ((mC + 3) / 4)),
                                                          (cl::size_type)mN * mH, 1};
        const std::array<cl::size_type, 3> region_output = {(cl::size_type)(mW * ((mC + 3) / 4)),
                                                          (cl::size_type)mN * mH, 1};
        mQueue.enqueueWriteImage(mInput, CL_TRUE, origin, region_input, 0, 0, input);

        auto t_end2 = std::chrono::high_resolution_clock::now();
        std::cout << "Wall time of H2D: "
                << std::chrono::duration<double, std::milli>(t_end2 - t_end1).count() << " ms" << std::endl;

        // 9. set kernel arguments
        mTestImage.setArg(0, mN);
        mTestImage.setArg(1, mH);
        mTestImage.setArg(2, mW);
        mTestImage.setArg(3, mC);
        mTestImage.setArg(4, mInput);
        mTestImage.setArg(5, mOutput);

        // 10. run the kernel
        cl::NDRange globalSize((mC + 3 ) / 4,
                               mW,
                               mN * mH);
        cl::NDRange localSize = cl::NullRange;
        mQueue.enqueueNDRangeKernel(mTestImage, cl::NullRange,
                globalSize, localSize);

        auto t_end3 = std::chrono::high_resolution_clock::now();
        std::cout << "Wall time of kernel: "
                << std::chrono::duration<double, std::milli>(t_end3 - t_end2).count() << " ms" << std::endl;

        // 11. get the result
        mQueue.enqueueReadImage(mOutput, CL_TRUE, origin, region_output, 0, 0, output);

        auto t_end4 = std::chrono::high_resolution_clock::now();
        std::cout << "Wall time of D2H: "
                << std::chrono::duration<double, std::milli>(t_end4 - t_end3).count() << " ms" << std::endl;

        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "Wall time of total OpenCL: "
                << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms" << std::endl;

	int i = 0;
	for (int n = 0; n < mN; n++)
        {
            for (int h = 0; h < mH; h++)
            {
                for (int w = 0; w < mW; w++)
                {
                    for (int c = 0; c < mC; c++)
                    {
                        std::cout << output[i] << "\t";
                        i++;
                    }
                }
		std::cout << std::endl;
            }
        }

    }

    return 0;
}
