/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/
/*
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 * Version: 2.08 beta
 *
 * Description:
 * Sample source code to illustrate how to deploy ResNet50 neural network
 * on DeePhi DPU platform.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <sstream>

/* header file OpenCV for image processing */
/#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;

/* Kernel name */
#define KERNEL_NAME_0 "caffe_resnet50_0"
/* Input Node  */
#define INPUT_NODE_0 "conv1"
/* Output Node */
#define OUTPUT_NODE_0 "fc1000"

void CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        result[i] = exp(data[i]);
        sum += result[i];
    }

    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, vector<string> &vkinds) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
        vkinds[ki.second].c_str());
        q.pop();
    }
}

int32_t str2int(const string &str) {
    stringstream ss;
    ss << str;
    int32_t result;
    ss >> result;
    return result;
}


void writefile (int8_t* data, size_t size, string filename){
    ofstream f_out(filename, ios::binary);
    f_out.write((char*)data, sizeof(char)*size);
    f_out.close();
}


int main(int argc, char* argv[]) {

    /* Create DPU kernel and task*/
    DPUKernel *kernel_0;
    DPUTask *task_0;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Load DPU Kernel */
    kernel_0 = dpuLoadKernel(KERNEL_NAME_0);

    /* Create DPU Task for ResNet50 */
    task_0 = dpuCreateTask(kernel_0, 0);

    //Load dump data and feed to DPU
    ifstream file_in(argv[1]);
    int8_t* data = dpuGetInputTensorAddress(task_0, INPUT_NODE_0);
    string s;
    while(getline(file_in, s)) {
	auto f = str2int(s);
	// cout << "transfor "<< f << endl;
	*data = (int8_t)f;
	data++;
    };
    file_in.close();

    dpuRunTask(task_0);

    /* Destroy DPU Task & free resources */
    dpuDestroyTask(task_0);

    /* Destroy DPU Kernel & free resources */
    dpuDestroyKernel(kernel_0);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
