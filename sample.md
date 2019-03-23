# Samples

## 1. Sample 

File Path: file_path

Comment Line Number: comment_end_line_number

Label: matched_label

```c++
comment_content
```

```c++
related_code
```

## 2. Sample 123

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/bgfg_KNN.cpp

Comment Line Number: 280

Label: var_def

```c++
/////////////////////////

```

```c++
protected:
    Size frameSize;
    int frameType;
    int nframes;
    
```

## 3. Sample 125

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/bgfg_KNN.cpp

Comment Line Number: 282

Label: empty

```c++
////////////////////////

```

```c++
    
```

## 4. Sample 149

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/bgfg_KNN.cpp

Comment Line Number: 359

Label: empty

```c++
// hold the offset

```

```c++
    
```

## 5. Sample 398

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/variational_refinement.cpp

Comment Line Number: 120

Label: empty

```c++
/* Parallelizing arbitrary operations with 3 input/output arguments */

```

```c++
    
```

## 6. Sample 415

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/variational_refinement.cpp

Comment Line Number: 669

Label: call

```c++
/* Add respective color constancy terms to the linear system coefficients: */

```

```c++
            weight = (delta2 / sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared)) / derivNorm;
            
```

## 7. Sample 534

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/dis_flow.cpp

Comment Line Number: 177

Label: var_def

```c++
//!< Gaussian pyramid for the y gradient of the current frame

```

```c++
    vector<UMat> u_I0ys; 
```

## 8. Sample 690

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/src/optflowgf.cpp

Comment Line Number: 452

Label: empty

```c++
// vertical blur

```

```c++
        
```

## 9. Sample 753

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/test/test_dxt.cpp

Comment Line Number: 4

Label: macro

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

```

```c++
#include "test_precomp.hpp"

```

## 10. Sample 800

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/test/test_eigen.cpp

Comment Line Number: 517

Label: var_def

```c++
// 1D Mat

```

```c++
    int N = GetParam();
    Mat_<double> srcZero = Mat_<double>::zeros(N, N);
    Mat_<double> expected_eigenvalueZero = Mat_<double>::zeros(N, 1);  
```

## 11. Sample 881

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/test/test_operations.cpp

Comment Line Number: 633

Label: empty

```c++
////////////////////////////////

```

```c++
        
```

## 12. Sample 990

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/test/test_mat.cpp

Comment Line Number: 356

Label: var_def

```c++
// eigenvectors have normalized length, but both directions v and -v are valid

```

```c++
        Mat r0 = rPCA.eigenvectors.row(i);
        Mat r1 = subEvec.row(i);
        
```

## 13. Sample 1123

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/test/test_math.cpp

Comment Line Number: 1423

Label: other

```c++
/*j*/
```

```c++
, int 
```

## 14. Sample 1493

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/perf/perf_convertTo.cpp

Comment Line Number: 41

Label: empty

```c++
// namespace

```

```c++
} 
```

## 15. Sample 1664

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/dxt.cpp

Comment Line Number: 3144

Label: empty

```c++
// otherwise reuse the tables calculated on the previous stage

```

```c++
            
```

## 16. Sample 1722

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/gl_core_3_1.cpp

Comment Line Number: 446

Label: empty

```c++
// Extension: 3.1

```

```c++
    
```

## 17. Sample 1757

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/gl_core_3_1.cpp

Comment Line Number: 2384

Label: empty

```c++
// Legacy

```

```c++
    
```

## 18. Sample 1760

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/gl_core_3_1.cpp

Comment Line Number: 2529

Label: empty

```c++
// Extension: 1.3

```

```c++
            
```

## 19. Sample 1834

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/hal_internal.cpp

Comment Line Number: 223

Label: if

```c++
//U stored in a

```

```c++
    if((flags & CV_HAL_SVD_MODIFY_A) && (flags & CV_HAL_SVD_FULL_UV)) 
```

## 20. Sample 1861

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/lda.cpp

Comment Line Number: 181

Label: var_def

```c++
// make sure the data has the correct shape

```

```c++
    int n = src.rows;
    int d = src.cols;
    
```

## 21. Sample 2877

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/ocl.cpp

Comment Line Number: 4165

Label: empty

```c++
// synchronized

```

```c++
    
```

## 22. Sample 2962

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/matrix_expressions.cpp

Comment Line Number: 23

Label: other

```c++
/*expr*/
```

```c++
    bool elementWise(const MatExpr& 
```

## 23. Sample 3414

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/matmul.simd.hpp

Comment Line Number: 2551

Label: macro

```c++
// namespace
```

```c++
#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} 
```

## 24. Sample 3509

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/softfloat.cpp

Comment Line Number: 494

Label: NOTHING

```c++
/*----------------------------------------------------------------------------
| Shifts 'a' right by the number of bits given in 'dist', which must be in
| the range 1 to 63.  If any nonzero bits are shifted off, they are "jammed"
| into the least-significant bit of the shifted value by setting the least-
| significant bit to 1.  This shifted-and-jammed value is returned.
*----------------------------------------------------------------------------*/

```

```c++

```

## 25. Sample 3514

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/softfloat.cpp

Comment Line Number: 534

Label: var_def

```c++
/*----------------------------------------------------------------------------
| A constant table that translates an 8-bit unsigned integer (the array index)
| into the number of leading 0 bits before the most-significant 1 of that
| integer.  For integer zero (index 0), the corresponding table element is 8.
*----------------------------------------------------------------------------*/

```

```c++
static const uint_least8_t softfloat_countLeadingZeros8[256] = {

```

## 26. Sample 3599

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/softfloat.cpp

Comment Line Number: 1729

Label: call

```c++
/*--------------------------------------------------------------------
        | Changing the shift of `rem' here requires also changing the initial
        | subtraction from `expDiff'.
        *--------------------------------------------------------------------*/

```

```c++
        recip32 = softfloat_approxRecip32_1( sigB>>21 );
        
```

## 27. Sample 3729

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/softfloat.cpp

Comment Line Number: 3469

Label: other

```c++
// 1.021897

```

```c++
    0x3ff059b0d3158574, 
```

## 28. Sample 3856

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/softfloat.cpp

Comment Line Number: 3657

Label: other

```c++
// 0.184922, 0.831169

```

```c++
    0x3fc7ab890210d909, 0x3fea98ef606a63be, 
```

## 29. Sample 4240

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/ocl_deprecated.hpp

Comment Line Number: 742

Label: func_def

```c++
/*
OCL_FUNC_P(cl_context, clCreateContextFromType,
    (const cl_context_properties * properties,
    cl_device_type device_type,
    void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
    void * user_data,
    cl_int * errcode_ret),
    (properties, device_type, pfn_notify, user_data, errcode_ret))

OCL_FUNC(cl_int, clGetContextInfo,
    (cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void * param_value,
    size_t * param_value_size_ret),
    (context, param_name, param_value_size,
    param_value, param_value_size_ret))
*/

```

```c++
OCL_FUNC(cl_int, clRetainContext, (cl_context context), (context))
OCL_FUNC_P(cl_command_queue, clCreateCommandQueue,
    (cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int * errcode_ret),
    (context, device, properties, errcode_ret))

```

## 30. Sample 4326

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/hal_replacement.hpp

Comment Line Number: 235

Label: comment

```c++
/**
Multiply: _dst[i] = scale * src1[i] * src2[i]_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param scale additional multiplier
*/

```

```c++
//! @addtogroup core_hal_interface_multiply Element-wise multiply
//! @{

```

## 31. Sample 4459

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/stitching/perf/perf_stich.cpp

Comment Line Number: 201

Label: empty

```c++
// SURF works just fine with default settings

```

```c++
        
```

## 32. Sample 4548

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/stitching/src/motion_estimators.cpp

Comment Line Number: 659

Label: empty

```c++
//     a b tx

```

```c++
        
```

## 33. Sample 4627

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/stitching/src/stitcher.cpp

Comment Line Number: 205

Label: empty

```c++
// Compensate exposure before finding seams

```

```c++
    
```

## 34. Sample 5005

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/test/test_filter.cpp

Comment Line Number: 2078

Label: empty

```c++
// should work like !BORDER_ISOLATED, so the function MUST read values in full matrix

```

```c++
    
```

## 35. Sample 5477

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/perf/perf_accumulate.cpp

Comment Line Number: 55

Label: NOTHING

```c++
///////////////////////////// AccumulateSquare ///////////////////////////////////

```

```c++

```

## 36. Sample 5531

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/perf/perf_corners.cpp

Comment Line Number: 4

Label: macro

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

```

```c++
#include "perf_precomp.hpp"

```

## 37. Sample 5866

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/color.simd_helpers.hpp

Comment Line Number: 21

Label: other

```c++
// == G2YF*16384

```

```c++
    G2Y = 9617, 
```

## 38. Sample 5874

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/color.simd_helpers.hpp

Comment Line Number: 159

Label: func_decl

```c++
// = delete;

```

```c++
    CvtColorLoop_Invoker(const CvtColorLoop_Invoker&);  
```

## 39. Sample 6089

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/demosaicing.cpp

Comment Line Number: 1452

Label: call

```c++
// GRs += {brow0[N6-1]; (srow[-bstep*2-1]+srow[-1])} * (T>gradNW)

```

```c++
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                
```

## 40. Sample 6120

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/min_enclosing_triangle.cpp

Comment Line Number: 366

Label: comment

```c++
//! Initialisation function

```

```c++
/*!
* @param triangle       Minimum area triangle enclosing the given polygon
* @param area           Area of the minimum area enclosing triangle
*/

```

## 41. Sample 6825

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/morph.simd.hpp

Comment Line Number: 54

Label: call

```c++
// forward declarations

```

```c++
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
Ptr<BaseRowFilter> getMorphologyRowFilter(int op, int type, int ksize, int anchor);
Ptr<BaseColumnFilter> getMorphologyColumnFilter(int op, int type, int ksize, int anchor);
Ptr<BaseFilter> getMorphologyFilter(int op, int type, const Mat& kernel, Point anchor);

```

## 42. Sample 7066

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/drawing.cpp

Comment Line Number: 1069

Label: NOTHING

```c++
/****************************************************************************************\
*                                Polygons filling                                        *
\****************************************************************************************/

```

```c++

```

## 43. Sample 7205

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/linefit.cpp

Comment Line Number: 315

Label: NOTHING

```c++
/* Takes an array of 2D points, type of distance (including user-defined
 distance specified by callbacks, fills the array of four floats with line
 parameters A, B, C, D, where (A, B) is the normalized direction vector,
 (C, D) is the point that belongs to the line. */

```

```c++

```

## 44. Sample 7235

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/imgwarp.sse4_1.cpp

Comment Line Number: 505

Label: empty

```c++
/* End of file. */

```

```c++
}

```

## 45. Sample 7578

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 1374

Label: empty

```c++
//Action_5: Assign label of block R

```

```c++
                                                                
```

## 46. Sample 7593

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 1482

Label: empty

```c++
//Action_11: Merge labels of block Q and S

```

```c++
                                                    
```

## 47. Sample 7652

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 1923

Label: empty

```c++
// Get rows pointer

```

```c++
                                
```

## 48. Sample 7769

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 2818

Label: empty

```c++
//Action_11: Merge labels of block Q and S

```

```c++
                                                    
```

## 49. Sample 7792

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 2977

Label: empty

```c++
//Action_15: Merge labels of block P, R and S

```

```c++
                                                            
```

## 50. Sample 7849

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 3382

Label: empty

```c++
//Action_12: Merge labels of block R and S

```

```c++
                                                                    
```

## 51. Sample 7858

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 3444

Label: empty

```c++
//Action_16: labels of block Q, R and S

```

```c++
                                                                        
```

## 52. Sample 7893

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/connectedcomponents.cpp

Comment Line Number: 3794

Label: empty

```c++
// Get rows pointer

```

```c++
                        
```

## 53. Sample 8101

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/distransform.cpp

Comment Line Number: 571

Label: func_decl

```c++
// stage 1: compute 1d distance transform of each column

```

```c++
    cv::AutoBuffer<uchar> _buf(std::max(m*2*sizeof(float) + (m*3+1)*sizeof(int), n*2*sizeof(float)));
    
```

## 54. Sample 8250

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/convhull.cpp

Comment Line Number: 244

Label: empty

```c++
// (except the exteme points).

```

```c++
                
```

## 55. Sample 8447

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/src/grabcut.cpp

Comment Line Number: 90

Label: var_def

```c++
/*mean*/
```

```c++
    const int modelSize = 3
```

## 56. Sample 8793

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/objdetect/src/hog.cpp

Comment Line Number: 739

Label: empty

```c++
// The detection algorithm runs in 4 nested loops (at each pyramid layer):

```

```c++
    
```

## 57. Sample 8989

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/test/gapi_fluid_test_kernels.cpp

Comment Line Number: 75

Label: empty

```c++
//std::cout << std::setw(4) << int(in_row[i]);

```

```c++
                
```

## 58. Sample 9142

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/test/test_main.cpp

Comment Line Number: 6

Label: NOTHING

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

```

```c++

```

## 59. Sample 9333

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/perf/perf_precomp.hpp

Comment Line Number: 27

Label: macro

```c++
// __OPENCV_GAPI_PERF_PRECOMP_HPP__

```

```c++
#endif 
```

## 60. Sample 9363

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ml/test/test_emknearestkmeans.cpp

Comment Line Number: 454

Label: empty

```c++
// train data

```

```c++
    
```

## 61. Sample 9607

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ml/src/data.cpp

Comment Line Number: 347

Label: empty

```c++
// maps for different variables if they are identical

```

```c++
        
```

## 62. Sample 10429

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgcodecs/src/grfmt_jpeg2000.hpp

Comment Line Number: 41

Label: NOTHING

```c++
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

```

```c++

```

## 63. Sample 10515

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgcodecs/src/rgbe.cpp

Comment Line Number: 135

Label: if

```c++
/*nonzero pixel*/

```

```c++
  if (rgbe[3]) {   
```

## 64. Sample 10595

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgcodecs/src/loadsave.cpp

Comment Line Number: 215

Label: empty

```c++
/// Open the file

```

```c++
    
```

## 65. Sample 10600

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgcodecs/src/loadsave.cpp

Comment Line Number: 310

Label: other

```c++
//0th row == visual top, 0th column == visual left-hand side

```

```c++
        case    IMAGE_ORIENTATION_TL: 
```

## 66. Sample 10791

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/test/test_darknet_importer.cpp

Comment Line Number: 326

Label: other

```c++
// a person

```

```c++
                                    1, 0,  0.980052f, 0.195856f, 0.378454f, 0.258626f, 0.629258f,  
```

## 67. Sample 11204

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/perf/perf_convolution.cpp

Comment Line Number: 197

Label: other

```c++
/* GFLOPS 0.095 x 1 = 0.095 */
```

```c++
,
    
```

## 68. Sample 11474

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/perf/perf_convolution.cpp

Comment Line Number: 467

Label: other

```c++
/* GFLOPS 0.003 x 1 = 0.003 */
```

```c++
,
    
```

## 69. Sample 11540

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/op_halide.cpp

Comment Line Number: 7

Label: NOTHING

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

```

```c++

```

## 70. Sample 11576

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/dnn.cpp

Comment Line Number: 567

Label: empty

```c++
// |       fp32 |        fp32 |

```

```c++
        
```

## 71. Sample 11682

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/dnn.cpp

Comment Line Number: 2114

Label: empty

```c++
// replace [conv]'s output blob to [eltwise]'s one

```

```c++
                                        
```

## 72. Sample 11889

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/test/test_video_io.cpp

Comment Line Number: 115

Label: empty

```c++
// Old Gstreamer are used in Ubuntu 14.04, so the following code could be removed after it's EOL

```

```c++
                
```

## 73. Sample 11896

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/test/test_video_io.cpp

Comment Line Number: 288

Label: empty

```c++
// the calculated frame count is also off by one. Ideally, we'd want

```

```c++
        
```

## 74. Sample 11982

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_pvapi.cpp

Comment Line Number: 159

Label: empty

```c++
//close();

```

```c++
    
```

## 75. Sample 12418

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/wrl.h

Comment Line Number: 202

Label: other

```c++
// E_ABORT

```

```c++
  case 0x80004004L: 
```

## 76. Sample 12476

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_openni2.cpp

Comment Line Number: 926

Label: call

```c++
// from mm to meters

```

```c++
                pointCloud_XYZ.at<cv::Point3f>(y, x) = cv::Point3f(worldX*0.001f, worldY*0.001f, worldZ*0.001f); 
```

## 77. Sample 12650

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_gstreamer.cpp

Comment Line Number: 1785

Label: other

```c++
/*val*/
```

```c++
, CV_OUT double* 
```

## 78. Sample 13073

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_ffmpeg_impl.hpp

Comment Line Number: 2587

Label: empty

```c++
// avoid FFMPEG warning 'clipping 1 dct coefficients...'

```

```c++
        
```

## 79. Sample 13087

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_ffmpeg_impl.hpp

Comment Line Number: 2707

Label: empty

```c++
// write the compressed frame in the media file

```

```c++
        
```

## 80. Sample 13235

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_dshow.cpp

Comment Line Number: 471

Label: empty

```c++
//directshow will try and get the closest possible framerate to what is requested

```

```c++
        
```

## 81. Sample 13725

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/src/cap_msmf.cpp

Comment Line Number: 1379

Label: var_def

```c++
// image format properties

```

```c++
    IAMVideoProcAmp *pProcAmp = NULL;
    IAMCameraControl *pProcControl = NULL;
    
```

## 82. Sample 13782

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/highgui/test/test_main.cpp

Comment Line Number: 4

Label: macro

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

```

```c++
#include "test_precomp.hpp"

```

## 83. Sample 14433

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/test/test_detectors_invariance.cpp

Comment Line Number: 18

Label: NOTHING

```c++
/*
 * Detector's rotation invariance check
 */

```

```c++

```

## 84. Sample 14557

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/test/test_detectors_invariance.impl.hpp

Comment Line Number: 4

Label: NOTHING

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

```

```c++

```

## 85. Sample 14583

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/test/test_orb.cpp

Comment Line Number: 82

Label: empty

```c++
// }

```

```c++
        
```

## 86. Sample 14784

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/brisk.cpp

Comment Line Number: 241

Label: var_def

```c++
// agast

```

```c++
  float scale_;
  float offset_;
  
```

## 87. Sample 14913

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/brisk.cpp

Comment Line Number: 1379

Label: empty

```c++
// interpolate the position:

```

```c++
        
```

## 88. Sample 15073

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/akaze.cpp

Comment Line Number: 121

Label: empty

```c++
// We use the random bit selection length binary descriptor

```

```c++
                    
```

## 89. Sample 15085

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/orb.cpp

Comment Line Number: 377

Label: other

```c++
/*mean (0), correlation (0)*/
```

```c++
    8,-3, 9,5
```

## 90. Sample 15381

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/hal_replacement.hpp

Comment Line Number: 40

Label: NOTHING

```c++
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

```

```c++

```

## 91. Sample 15550

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 80

Label: NOTHING

```c++
//
// The Google C++ Testing and Mocking Framework (Google Test)

```

```c++

```

## 92. Sample 15771

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 1052

Label: empty

```c++
//   type_param:     the name of the test's type parameter, or NULL if

```

```c++
  
```

## 93. Sample 15826

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 1190

Label: var_def

```c++
// GTEST_HAS_DEATH_TEST

```

```c++
  friend class ReplaceDeathTestFactory;
#endif  
```

## 94. Sample 15889

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 1320

Label: empty

```c++
// The time of the test program start, in ms from the start of the

```

```c++
  
```

## 95. Sample 16045

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 2937

Label: empty

```c++
// namespace internal

```

```c++
}  
```

## 96. Sample 16241

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 4018

Label: call

```c++
// Constructs a TestInfo object. It assumes ownership of the test factory
// object.

```

```c++
TestInfo::TestInfo(const std::string& a_test_case_name,
                   const std::string& a_name,
                   const char* a_type_param,
                   const char* a_value_param,
                   internal::CodeLocation a_code_location,
                   internal::TypeId fixture_class_id,
                   internal::TestFactoryBase* factory)
    : test_case_name_(a_test_case_name),
      name_(a_name),
      type_param_(a_type_param ? new std::string(a_type_param) : NULL),
      value_param_(a_value_param ? new std::string(a_value_param) : NULL),
      location_(a_code_location),
      fixture_class_id_(fixture_class_id),
      should_run_(false),
      is_disabled_(false),
      matches_filter_(false),
      factory_(factory),
      result_() {
```

## 97. Sample 16461

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 6001

Label: call

```c++
// Gets the number of successful test cases.

```

```c++
int UnitTest::successful_test_case_count() const {

```

## 98. Sample 16959

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/src/ts_gtest.cpp

Comment Line Number: 8822

Label: macro

```c++
// GTEST_OS_LINUX

```

```c++
#  if GTEST_OS_LINUX
  GTEST_DEATH_TEST_CHECK_SYSCALL_(
      sigaction(SIGPROF, &saved_sigprof_action, NULL));
#  endif  
```

## 99. Sample 17352

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/photo/test/test_denoising.cpp

Comment Line Number: 41

Label: NOTHING

```c++
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

```

```c++

```

## 100. Sample 17733

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/test/test_undistort.cpp

Comment Line Number: 1134

Label: empty

```c++
//useCPlus = ((cvtest::randInt(rng) % 2)!=0);

```

```c++
    
```

## 101. Sample 17969

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/test/test_calibration_hand_eye.cpp

Comment Line Number: 151

Label: empty

```c++
//Maybe a better accuracy test would be to compare the mean and std errors with some thresholds?

```

```c++
                
```

## 102. Sample 18033

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/test/test_cameracalibration.cpp

Comment Line Number: 607

Label: empty

```c++
//only for c-version of test (it does not provides evaluation of perViewErrors

```

```c++
            
```

## 103. Sample 18359

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/calibration.cpp

Comment Line Number: 592

Label: func_decl

```c++
//        _m = cvCreateMat( 1, count, CV_64FC2 );

```

```c++
        CV_Error( CV_StsBadArg, "Homogeneous coordinates are not supported" );
    }

```

## 104. Sample 18504

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/triangulate.cpp

Comment Line Number: 94

Label: for

```c++
/* For each point */

```

```c++
    for( int i = 0; i < numPoints; i++ )
```

## 105. Sample 18657

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/dls.cpp

Comment Line Number: 362

Label: other

```c++
// s2 * s3^2

```

```c++
    f2coeff[12] = 4*D[1][1] - 4*D[5][5] + 4*D[5][9] + 8*D[6][6] + 8*D[6][8] + 8*D[8][6] + 8*D[8][8] + 4*D[9][5] - 4*D[9][9]; 
```

## 106. Sample 18740

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/chessboard.hpp

Comment Line Number: 217

Label: empty

```c++
// using 1D homography

```

```c++
                
```

## 107. Sample 18841

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/ptsetreg.cpp

Comment Line Number: 528

Label: empty

```c++
// we need 3 points to estimate affine transform

```

```c++
        
```

## 108. Sample 18890

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/calibinit.cpp

Comment Line Number: 278

Label: call

```c++
//COMPUTE INTENSITY HISTOGRAM OF INPUT IMAGE

```

```c++
template<typename ArrayContainer>
static void icvGetIntensityHistogram256(const Mat& img, ArrayContainer& piHist)
{

```

## 109. Sample 19124

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/stereosgbm.cpp

Comment Line Number: 288

Label: other

```c++
/*
 computes disparity for "roi" in img1 w.r.t. img2 and write it to disp1buf.
 that is, disp1buf(x, y)=d means that img1(x+roi.x, y+roi.y) ~ img2(x+roi.x-d, y+roi.y).
 minD <= d < maxD.
 disp2full is the reverse disparity map, that is:
 disp2full(x+roi.x,y+roi.y)=d means that img2(x+roi.x, y+roi.y) ~ img1(x+roi.x+d, y+roi.y)

 note that disp1buf will have the same size as the roi and
 disp2full will have the same size as img1 (or img2).
 On exit disp2buf is not the final disparity, it is an intermediate result that becomes
 final after all the tiles are processed.

 the disparity in disp1buf is written with sub-pixel accuracy
 (4 fractional bits, see StereoSGBM::DISP_SCALE),
 using quadratic interpolation, while the disparity in disp2buf
 is written as is, without interpolation.

 disp2cost also has the same size as img1 (or img2).
 It contains the minimum current cost, used to find the best disparity, corresponding to the minimal cost.
 */

```

```c++
static void computeDisparitySGBM( const Mat& img1, const Mat& img2,
                                 Mat& disp1, const StereoSGBMParams& params,
                                 Mat& buffer )
{

```

## 110. Sample 19319

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/calibration_handeye.cpp

Comment Line Number: 335

Label: empty

```c++
//Hgi is from Gi (gripper) to RW (robot base)

```

```c++
            
```

## 111. Sample 19363

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/solvepnp.cpp

Comment Line Number: 399

Label: if

```c++
// inliers mask

```

```c++
            if((int)_mask_local_inliers.at<uchar>(i) != 0) 
```

## 112. Sample 19506

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/chessboard.cpp

Comment Line Number: 2545

Label: empty

```c++
// trace contour

```

```c++
    
```

## 113. Sample 19526

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/chessboard.cpp

Comment Line Number: 2825

Label: call

```c++
// || pt.response < noise)

```

```c++
        cv::KeyPoint &pt = *iter1;
        const std::vector<float> &angles_i3 = *iter2;
        if(angles_i3.size() != 2)
```

## 114. Sample 19630

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/fisheye.cpp

Comment Line Number: 202

Label: empty

```c++
//double inv_r = r > 1e-8 ? 1.0/r : 1;

```

```c++
            
```

## 115. Sample 19900

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/rho.cpp

Comment Line Number: 219

Label: other

```c++
/* Works:    2000 */

```

```c++
                                  unsigned       rConvg,  
```

## 116. Sample 19904

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/rho.cpp

Comment Line Number: 223

Label: other

```c++
/* Works:       0 */

```

```c++
                                  unsigned       flags,   
```

## 117. Sample 20089

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/rho.cpp

Comment Line Number: 1553

Label: empty

```c++
/**
     * Randomized RANSAC with Sequential Probability Ratio Test, ICCV 2005
     * Eq (2)
     */

```

```c++
    
```

## 118. Sample 20208

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/src/rho.cpp

Comment Line Number: 2433

Label: other

```c++
/* Lij = ... / Ljj */

```

```c++
            L[i][j] = x / L[j][j];     
```

## 119. Sample 20735

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core.hpp

Comment Line Number: 3084

Label: empty

```c++
/** @brief Stores algorithm parameters in a file storage
    */

```

```c++
    
```

## 120. Sample 20830

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/perf/opencl/perf_arithm.cpp

Comment Line Number: 1066

Label: NOTHING

```c++
///////////// Transform ////////////////////////

```

```c++

```

## 121. Sample 20851

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/utils/datafile.cpp

Comment Line Number: 372

Label: func_decl

```c++
// not found

```

```c++
    return cv::String();  
```

## 122. Sample 20916

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/test/ocl/test_filter2d.cpp

Comment Line Number: 141

Label: empty

```c++
// namespace opencv_test::ocl

```

```c++
} } 
```

## 123. Sample 20932

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/test/ocl/test_warp.cpp

Comment Line Number: 288

Label: var_def

```c++
// Make sure the width is a multiple of the requested value, and no more

```

```c++
        Size srcRoiSize = randomSize(10, MAX_VALUE), dstRoiSize;
        
```

## 124. Sample 20960

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/test/ocl/test_filters.cpp

Comment Line Number: 172

Label: other

```c++
// border type

```

```c++
                BorderType, 
```

## 125. Sample 21177

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/include/opencv2/imgproc.hpp

Comment Line Number: 275

Label: other

```c++
/** \brief Specify the polar mapping mode
@sa warpPolar
*/

```

```c++
enum WarpPolarMode
{

```

## 126. Sample 21291

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/include/opencv2/imgproc.hpp

Comment Line Number: 675

Label: other

```c++
//COLOR_YUV2RGB_VYUY = 109,

```

```c++
    COLOR_YUV2RGB_UYVY = 107,
    COLOR_YUV2BGR_UYVY = 108,
    
```

## 127. Sample 21531

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/include/opencv2/imgproc.hpp

Comment Line Number: 3643

Label: other

```c++
//!< \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]

```

```c++
    TM_CCORR_NORMED  = 3, 
```

## 128. Sample 22142

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/test/internal/gapi_int_island_fusion_tests.cpp

Comment Line Number: 46

Label: empty

```c++
// Inspect the graph and verify the islands configuration

```

```c++
    
```

## 129. Sample 22342

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/test/internal/gapi_int_island_tests.cpp

Comment Line Number: 197

Label: empty

```c++
// (in0) -> Not  -> (tmp0) --> Add ---------> (tmp2) --> AddC -------> (out0)

```

```c++
    
```

## 130. Sample 22398

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/test/internal/gapi_int_island_tests.cpp

Comment Line Number: 352

Label: empty

```c++
//                :......................................................:

```

```c++
    
```

## 131. Sample 22503

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/test/common/gapi_imgproc_tests_inl.hpp

Comment Line Number: 46

Label: empty

```c++
// Comparison //////////////////////////////////////////////////////////////

```

```c++
    
```

## 132. Sample 22978

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/perf/common/gapi_core_perf_tests_inl.hpp

Comment Line Number: 1275

Label: empty

```c++
// OpenCV code ///////////////////////////////////////////////////////////

```

```c++
    
```

## 133. Sample 23014

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/perf/common/gapi_core_perf_tests_inl.hpp

Comment Line Number: 1567

Label: empty

```c++
// OpenCV code ///////////////////////////////////////////////////////////

```

```c++
    
```

## 134. Sample 23224

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/api/gscalar.cpp

Comment Line Number: 66

Label: macro

```c++
// !defined(GAPI_STANDALONE)

```

```c++
#endif 
```

## 135. Sample 23367

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/api/gcomputation.cpp

Comment Line Number: 195

Label: empty

```c++
// Island must have a printable name.

```

```c++
        
```

## 136. Sample 23427

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/executor/gexecutor.hpp

Comment Line Number: 96

Label: empty

```c++
// namespace cv

```

```c++
} 
```

## 137. Sample 23650

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/compiler/gmodel.hpp

Comment Line Number: 236

Label: empty

```c++
// Array is sparse, as metadata for non-gapi input objects is empty

```

```c++
    
```

## 138. Sample 23741

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ml/include/opencv2/ml.hpp

Comment Line Number: 329

Label: empty

```c++
/** @brief Returns the number of variables in training samples */

```

```c++
    
```

## 139. Sample 24212

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.cc

Comment Line Number: 1019

Label: empty

```c++
// optional bytes s = 4;

```

```c++
      
```

## 140. Sample 24348

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.cc

Comment Line Number: 2260

Label: if

```c++
/* 50 & 0xFF */
```

```c++
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(50u 
```

## 141. Sample 24385

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.cc

Comment Line Number: 2606

Label: if

```c++
// @@protoc_insertion_point(class_specific_copy_from_start:opencv_onnx.NodeProto)

```

```c++
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

```

## 142. Sample 24601

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.cc

Comment Line Number: 4754

Label: if

```c++
/* 66 & 0xFF */
```

```c++
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(66u 
```

## 143. Sample 25021

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.h

Comment Line Number: 2850

Label: func_decl

```c++
// @@protoc_insertion_point(field_set:opencv_onnx.AttributeProto.type)

```

```c++
  assert(::opencv_onnx::AttributeProto_AttributeType_IsValid(value));
  set_has_type();
  type_ = value;
  
```

## 144. Sample 25433

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.h

Comment Line Number: 5470

Label: NOTHING

```c++
// -------------------------------------------------------------------

```

```c++

```

## 145. Sample 25483

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/onnx/opencv-onnx.pb.h

Comment Line Number: 5803

Label: NOTHING

```c++
// -------------------------------------------------------------------

```

```c++

```

## 146. Sample 25645

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 2865

Label: empty

```c++
// optional .opencv_caffe.NetState state = 6;

```

```c++
  
```

## 147. Sample 25876

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 6004

Label: call

```c++
// -------------------------------------------------------------------

```

```c++
  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  bool use_global_stats_;
  bool scale_bias_;
  float moving_average_fraction_;
  float eps_;
  friend struct ::protobuf_opencv_2dcaffe_2eproto::TableStruct;
  friend void ::protobuf_opencv_2dcaffe_2eproto::InitDefaultsBatchNormParameterImpl();
}
```

## 148. Sample 25931

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 6741

Label: func_decl

```c++
// FOR INTERNAL USE ONLY

```

```c++
  static void InitAsDefaultInstance();  
```

## 149. Sample 25959

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 7126

Label: empty

```c++
// implements Message ----------------------------------------------

```

```c++
  
```

## 150. Sample 26054

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 8473

Label: empty

```c++
// nested types ----------------------------------------------------

```

```c++
  
```

## 151. Sample 26064

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 8621

Label: empty

```c++
// optional .opencv_caffe.HingeLossParameter.Norm norm = 1 [default = L1];

```

```c++
  
```

## 152. Sample 26241

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 11098

Label: empty

```c++
// implements Message ----------------------------------------------

```

```c++
  
```

## 153. Sample 26255

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 11279

Label: empty

```c++
// optional .opencv_caffe.FillerParameter bias_filler = 5;

```

```c++
  
```

## 154. Sample 26764

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 16477

Label: NOTHING

```c++
// -------------------------------------------------------------------

```

```c++

```

## 155. Sample 26802

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 16688

Label: empty

```c++
// @@protoc_insertion_point(field_get:opencv_caffe.NetParameter.force_backward)

```

```c++
  
```

## 156. Sample 26878

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 17223

Label: other

```c++
// @@protoc_insertion_point(field_set_allocated:opencv_caffe.SolverParameter.train_state)

```

```c++
  train_state_ = train_state;
  
```

## 157. Sample 26924

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 17541

Label: call

```c++
// optional float gamma = 9;

```

```c++
inline bool SolverParameter::has_gamma() const {

```

## 158. Sample 27203

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 19306

Label: var_def

```c++
// @@protoc_insertion_point(field_get:opencv_caffe.LayerParameter.transform_param)

```

```c++
  const ::opencv_caffe::TransformationParameter* p = transform_param_;
  
```

## 159. Sample 27224

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 19527

Label: empty

```c++
// @@protoc_insertion_point(field_release:opencv_caffe.LayerParameter.batch_norm_param)

```

```c++
  
```

## 160. Sample 27799

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 24446

Label: call

```c++
// @@protoc_insertion_point(field_add:opencv_caffe.DummyDataParameter.channels)

```

```c++
  channels_.Add(value);
  
```

## 161. Sample 27872

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 24939

Label: func_decl

```c++
// @@protoc_insertion_point(field_set:opencv_caffe.FlattenParameter.end_axis)

```

```c++
  set_has_end_axis();
  end_axis_ = value;
  
```

## 162. Sample 28023

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 25996

Label: func_decl

```c++
// @@protoc_insertion_point(field_set:opencv_caffe.LRNParameter.alpha)

```

```c++
  set_has_alpha();
  alpha_ = value;
  
```

## 163. Sample 28618

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.h

Comment Line Number: 30670

Label: empty

```c++
// @@protoc_insertion_point(field_release:opencv_caffe.V0LayerParameter.type)

```

```c++
  
```

## 164. Sample 29076

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 2571

Label: other

```c++
// no _weak_field_map_

```

```c++
  ~0u,  
```

## 165. Sample 29368

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 5844

Label: empty

```c++
// repeated .opencv_caffe.BlobProto blobs = 1;

```

```c++
  
```

## 166. Sample 29591

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 7741

Label: empty

```c++
// @@protoc_insertion_point(serialize_start:opencv_caffe.DetectionOutputParameter)

```

```c++
  
```

## 167. Sample 29924

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 10669

Label: empty

```c++
// optional float momentum = 11;

```

```c++
  
```

## 168. Sample 30257

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 14322

Label: empty

```c++
// optional .opencv_caffe.ConvolutionParameter convolution_param = 106;

```

```c++
      
```

## 169. Sample 30275

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 14430

Label: empty

```c++
// optional .opencv_caffe.ImageDataParameter image_data_param = 115;

```

```c++
      
```

## 170. Sample 30283

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 14478

Label: empty

```c++
// optional .opencv_caffe.MemoryDataParameter memory_data_param = 119;

```

```c++
      
```

## 171. Sample 30401

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 15218

Label: empty

```c++
// optional .opencv_caffe.LogParameter log_param = 134;

```

```c++
  
```

## 172. Sample 30444

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 15506

Label: empty

```c++
// optional .opencv_caffe.DummyDataParameter dummy_data_param = 109;

```

```c++
  
```

## 173. Sample 30590

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 16915

Label: empty

```c++
// optional uint32 crop_size = 3 [default = 0];

```

```c++
  
```

## 174. Sample 30654

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 17438

Label: func_decl

```c++
// @@protoc_insertion_point(class_specific_merge_from_start:opencv_caffe.LossParameter)

```

```c++
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

```

## 175. Sample 30737

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 18226

Label: other

```c++
// Prevent compiler warnings about cached_has_bits being unused

```

```c++
  ::google::protobuf::uint32 cached_has_bits = 0;
  
```

## 176. Sample 30811

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 18890

Label: other

```c++
// Prevent compiler warnings about cached_has_bits being unused

```

```c++
  ::google::protobuf::uint32 cached_has_bits = 0;
  
```

## 177. Sample 31014

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 20576

Label: var_def

```c++
// @@protoc_insertion_point(message_byte_size_start:opencv_caffe.CropParameter)

```

```c++
  size_t total_size = 0;

```

## 178. Sample 31332

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 23465

Label: empty

```c++
// @@protoc_insertion_point(generalized_merge_from_cast_fail:opencv_caffe.ELUParameter)

```

```c++
  
```

## 179. Sample 31699

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 26813

Label: func_decl

```c++
// @@protoc_insertion_point(generalized_merge_from_start:opencv_caffe.InnerProductParameter)

```

```c++
  GOOGLE_DCHECK_NE(&from, this);
  const InnerProductParameter* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const InnerProductParameter>(
          &from);
  if (source == NULL) {

```

## 180. Sample 31714

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 26983

Label: if

```c++
/* 10 & 0xFF */
```

```c++
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u 
```

## 181. Sample 31756

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 27341

Label: empty

```c++
// optional float scale = 2 [default = 1];

```

```c++
  
```

## 182. Sample 31842

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 28109

Label: empty

```c++
// optional uint32 height = 3;

```

```c++
  
```

## 183. Sample 31901

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 28619

Label: macro

```c++
// !defined(_MSC_VER) || _MSC_VER >= 1900

```

```c++
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int ParameterParameter::kShapeFieldNumber;
#endif  
```

## 184. Sample 32117

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 30566

Label: empty

```c++
// optional bool debug_info = 4 [default = false];

```

```c++
  
```

## 185. Sample 32119

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 30580

Label: empty

```c++
// @@protoc_insertion_point(serialize_end:opencv_caffe.RecurrentParameter)

```

```c++
  
```

## 186. Sample 32172

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 31027

Label: empty

```c++
// @@protoc_insertion_point(generalized_merge_from_cast_fail:opencv_caffe.ReductionParameter)

```

```c++
  
```

## 187. Sample 32343

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 32611

Label: empty

```c++
// @@protoc_insertion_point(serialize_to_array_start:opencv_caffe.SliceParameter)

```

```c++
  
```

## 188. Sample 32611

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 35523

Label: if

```c++
/* 18 & 0xFF */
```

```c++
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u 
```

## 189. Sample 32928

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 38116

Label: var_def

```c++
// @@protoc_insertion_point(parse_failure:opencv_caffe.V0LayerParameter)

```

```c++
  return true;
failure:
  
```

## 190. Sample 32947

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 38231

Label: empty

```c++
// optional string meanfile = 18;

```

```c++
  
```

## 191. Sample 32984

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 38450

Label: empty

```c++
// optional float alpha = 14 [default = 1];

```

```c++
  
```

## 192. Sample 33062

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 39203

Label: macro

```c++
// @@protoc_insertion_point(parse_start:opencv_caffe.PReLUParameter)

```

```c++
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  
```

## 193. Sample 33088

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 39422

Label: macro

```c++
// !defined(_MSC_VER) || _MSC_VER >= 1900

```

```c++
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int NormalizedBBox::kXminFieldNumber;
const int NormalizedBBox::kYminFieldNumber;
const int NormalizedBBox::kXmaxFieldNumber;
const int NormalizedBBox::kYmaxFieldNumber;
const int NormalizedBBox::kLabelFieldNumber;
const int NormalizedBBox::kDifficultFieldNumber;
const int NormalizedBBox::kScoreFieldNumber;
const int NormalizedBBox::kSizeFieldNumber;
#endif  
```

## 194. Sample 33246

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 40682

Label: empty

```c++
// @@protoc_insertion_point(generalized_merge_from_cast_success:opencv_caffe.ProposalParameter)

```

```c++
  
```

## 195. Sample 33272

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/caffe/opencv-caffe.pb.cc

Comment Line Number: 40951

Label: empty

```c++
// @@protoc_insertion_point(serialize_to_array_start:opencv_caffe.PSROIPoolingParameter)

```

```c++
  
```

## 196. Sample 33338

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/types.pb.h

Comment Line Number: 49

Label: empty

```c++
// namespace opencv_tensorflow

```

```c++
}  
```

## 197. Sample 33354

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/tensor.pb.h

Comment Line Number: 38

Label: struct

```c++
// Internal implementation detail -- do not use these members.

```

```c++
struct TableStruct {

```

## 198. Sample 33538

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/attr_value.pb.cc

Comment Line Number: 149

Label: other

```c++
// no _weak_field_map_

```

```c++
  ~0u,  
```

## 199. Sample 33648

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/attr_value.pb.cc

Comment Line Number: 1333

Label: empty

```c++
// .opencv_tensorflow.AttrValue.ListValue list = 1;

```

```c++
  
```

## 200. Sample 33669

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/attr_value.pb.cc

Comment Line Number: 1457

Label: empty

```c++
// string placeholder = 9;

```

```c++
  
```

## 201. Sample 33684

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/attr_value.pb.cc

Comment Line Number: 1576

Label: empty

```c++
// @@protoc_insertion_point(generalized_merge_from_cast_fail:opencv_tensorflow.AttrValue)

```

```c++
  
```

## 202. Sample 33748

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/function.pb.h

Comment Line Number: 334

Label: empty

```c++
// nested types ----------------------------------------------------

```

```c++
  
```

## 203. Sample 33856

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/function.pb.h

Comment Line Number: 1184

Label: call

```c++
// repeated .opencv_tensorflow.FunctionDef.Node node = 2;

```

```c++
inline int FunctionDef::node_size() const {

```

## 204. Sample 33872

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/function.pb.h

Comment Line Number: 1271

Label: call

```c++
// @@protoc_insertion_point(field_set_allocated:opencv_tensorflow.GradientDef.function_name)

```

```c++
  function_name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), function_name,
      GetArenaNoVirtual());
  
```

## 205. Sample 34057

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/op_def.pb.cc

Comment Line Number: 1710

Label: empty

```c++
// repeated .opencv_tensorflow.OpDef.ArgDef output_arg = 3;

```

```c++
      
```

## 206. Sample 34373

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/tensor_shape.pb.cc

Comment Line Number: 639

Label: empty

```c++
// @@protoc_insertion_point(serialize_end:opencv_tensorflow.TensorShapeProto)

```

```c++
  
```

## 207. Sample 34382

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/tensor_shape.pb.cc

Comment Line Number: 703

Label: func_decl

```c++
// @@protoc_insertion_point(generalized_merge_from_start:opencv_tensorflow.TensorShapeProto)

```

```c++
  GOOGLE_DCHECK_NE(&from, this);
  const TensorShapeProto* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const TensorShapeProto>(
          &from);
  if (source == NULL) {

```

## 208. Sample 34836

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/op_def.pb.h

Comment Line Number: 2309

Label: call

```c++
// string explanation = 2;

```

```c++
inline void OpDeprecation::clear_explanation() {

```

## 209. Sample 35124

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/graph.pb.h

Comment Line Number: 185

Label: empty

```c++
// .opencv_tensorflow.FunctionDefLibrary library = 2;

```

```c++
  
```

## 210. Sample 35145

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/graph.pb.h

Comment Line Number: 494

Label: empty

```c++
// @@protoc_insertion_point(field_get:opencv_tensorflow.GraphDef.node)

```

```c++
  
```

## 211. Sample 35151

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/misc/tensorflow/graph.pb.h

Comment Line Number: 522

Label: var_def

```c++
// @@protoc_insertion_point(field_get:opencv_tensorflow.GraphDef.versions)

```

```c++
  const ::opencv_tensorflow::VersionDef* p = versions_;
  
```

## 212. Sample 35435

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/layers/shuffle_channel_layer.cpp

Comment Line Number: 4

Label: NOTHING

```c++
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

```

```c++

```

## 213. Sample 35467

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/layers/recurrent_layers.cpp

Comment Line Number: 299

Label: func_decl

```c++
//+b

```

```c++
            gemm(dummyOnes, bias, 1, gates, 1, gates);          
```

## 214. Sample 35850

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/torch/torch_importer.cpp

Comment Line Number: 320

Label: func_decl

```c++
//value

```

```c++
                readObject(); 
```

## 215. Sample 36023

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/tensorflow/tf_importer.cpp

Comment Line Number: 792

Label: func_decl

```c++
//  N    C    W    H

```

```c++
                    std::swap(paddings.at<int32_t>(2), paddings.at<int32_t>(6));
                    std::swap(paddings.at<int32_t>(3), paddings.at<int32_t>(7));
                    
```

## 216. Sample 36029

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/tensorflow/tf_importer.cpp

Comment Line Number: 827

Label: empty

```c++
// For the object detection networks, TensorFlow Object Detection API

```

```c++
            
```

## 217. Sample 36040

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/tensorflow/tf_importer.cpp

Comment Line Number: 966

Label: else

```c++
// is a vector

```

```c++
                else  
```

## 218. Sample 36353

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/include/opencv2/videoio.hpp

Comment Line Number: 275

Label: NOTHING

```c++
//! @} OpenNI

```

```c++

```

## 219. Sample 36400

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/include/opencv2/videoio.hpp

Comment Line Number: 344

Label: other

```c++
//!< Selects camera signalling LED.

```

```c++
       CAP_PROP_XI_LED_SELECTOR                                 = 411, 
```

## 220. Sample 36765

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/highgui/include/opencv2/highgui.hpp

Comment Line Number: 700

Label: var_def

```c++
//!< PointSize

```

```c++
    int         line_type; 
```

## 221. Sample 36863

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/include/opencv2/features2d.hpp

Comment Line Number: 894

Label: empty

```c++
/** @brief Returns true if the descriptor matcher supports masking permissible matches.
     */

```

```c++
    
```

## 222. Sample 36984

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/kaze/AKAZEFeatures.cpp

Comment Line Number: 429

Label: other

```c++
/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param image Input image for which the nonlinear scale space needs to be created
 */

```

```c++
template<typename MatType>
static inline void
create_nonlinear_scale_space(InputArray image, const AKAZEOptions &options,
  const std::vector<std::vector<float > > &tsteps_evolution, std::vector<Evolution<MatType> > &evolution)
{

```

## 223. Sample 37051

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/kaze/AKAZEFeatures.cpp

Comment Line Number: 903

Label: empty

```c++
// Compute the gradient

```

```c++
        
```

## 224. Sample 37061

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/kaze/AKAZEFeatures.cpp

Comment Line Number: 1185

Label: func_def

```c++
/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 * @param desc Matrix to store the descriptors
 */

```

```c++
void AKAZEFeatures::Compute_Descriptors(std::vector<KeyPoint>& kpts, OutputArray descriptors)
{

```

## 225. Sample 37198

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/kaze/KAZEFeatures.cpp

Comment Line Number: 336

Label: empty

```c++
// Now fill the vector of keypoints!!!

```

```c++
    
```

## 226. Sample 37406

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/features2d/src/kaze/TEvolution.h

Comment Line Number: 29

Label: var_def

```c++
///< Evolution image

```

```c++
  Mat Lt;               
```

## 227. Sample 37498

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts.hpp

Comment Line Number: 485

Label: empty

```c++
// unexpected response on passing bad arguments to the tested function

```

```c++
        
```

## 228. Sample 37565

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/photo/include/opencv2/photo.hpp

Comment Line Number: 482

Label: empty

```c++
/** @brief Short version of process, that doesn't take extra arguments.

    @param src vector of input images
    @param dst vector of aligned images
     */

```

```c++
    
```

## 229. Sample 37588

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/photo/include/opencv2/photo.hpp

Comment Line Number: 705

Label: NOTHING

```c++
//! @} photo_decolor

```

```c++

```

## 230. Sample 37698

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/include/opencv2/calib3d.hpp

Comment Line Number: 1931

Label: call

```c++
/** @brief Recover relative camera rotation and translation from an estimated essential matrix and the
corresponding points in two images, using cheirality check. Returns the number of inliers which pass
the check.

@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera matrix.
@param R Recovered relative rotation.
@param t Recovered relative translation.
@param mask Input/output mask for inliers in points1 and points2.
:   If it is not empty, then it marks inliers in points1 and points2 for then given essential
matrix E. Only these inliers will be used to recover pose. In the output mask only inliers
which pass the cheirality check.
This function decomposes an essential matrix using decomposeEssentialMat and then verifies possible
pose hypotheses by doing cheirality check. The cheirality check basically means that the
triangulated 3D points should have positive depth. Some details can be found in @cite Nister03 .

This function can be used to process output E and mask from findEssentialMat. In this scenario,
points1 and points2 are the same input for findEssentialMat. :
@code
    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    // cametra matrix with both focal lengths = 1, and principal point = (0, 0)
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    Mat E, R, t, mask;

    E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
@endcode
 */

```

```c++
CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            InputArray cameraMatrix, OutputArray R, OutputArray t,
                            InputOutputArray mask = noArray() );

```

## 231. Sample 37713

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/calib3d/include/opencv2/calib3d.hpp

Comment Line Number: 2317

Label: other

```c++
/** @brief Decompose a homography matrix to rotation(s), translation(s) and plane normal(s).

@param H The input homography matrix between two images.
@param K The input intrinsic camera calibration matrix.
@param rotations Array of rotation matrices.
@param translations Array of translation matrices.
@param normals Array of plane normal matrices.

This function extracts relative camera motion between two views observing a planar object from the
homography H induced by the plane. The intrinsic camera matrix K must also be provided. The function
may return up to four mathematical solution sets. At least two of the solutions may further be
invalidated if point correspondences are available by applying positive depth constraint (all points
must be in front of the camera). The decomposition method is described in detail in @cite Malis .
 */

```

```c++
CV_EXPORTS_W int decomposeHomographyMat(InputArray H,
                                        InputArray K,
                                        OutputArrayOfArrays rotations,
                                        OutputArrayOfArrays translations,
                                        OutputArrayOfArrays normals);

```

## 232. Sample 37848

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/video/include/opencv2/video/background_segm.hpp

Comment Line Number: 171

Label: empty

```c++
/** @brief Returns the shadow detection flag

    If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorMOG2 for
    details.
     */

```

```c++
    
```

## 233. Sample 37983

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/flann/include/opencv2/flann/kdtree_index.h

Comment Line Number: 423

Label: empty

```c++
/**
     * Performs an exact nearest neighbor search. The exact search performs a full
     * traversal of the tree.
     */

```

```c++
    
```

## 234. Sample 38326

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/flann/include/opencv2/flann/lsh_table.h

Comment Line Number: 113

Label: empty

```c++
// Display the histogram

```

```c++
    
```

## 235. Sample 38393

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/flann/include/opencv2/flann/lsh_table.h

Comment Line Number: 478

Label: empty

```c++
// TODO compute mean and std

```

```c++
    
```

## 236. Sample 38408

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/flann/include/opencv2/flann/dynamic_bitset.h

Comment Line Number: 99

Label: empty

```c++
/** @brief set one bit to 0
     * @param index
     */

```

```c++
    
```

## 237. Sample 38883

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/softfloat.hpp

Comment Line Number: 83

Label: comment

```c++
/** @addtogroup core_utils_softfloat

  [SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) is a software implementation
  of floating-point calculations according to IEEE 754 standard.
  All calculations are done in integers, that's why they are machine-independent and bit-exact.
  This library can be useful in accuracy-critical parts like look-up tables generation, tests, etc.
  OpenCV contains a subset of SoftFloat partially rewritten to C++.

  ### Types

  There are two basic types: @ref softfloat and @ref softdouble.
  These types are binary compatible with float and double types respectively
  and support conversions to/from them.
  Other types from original SoftFloat library like fp16 or fp128 were thrown away
  as well as quiet/signaling NaN support, on-the-fly rounding mode switch
  and exception flags (though exceptions can be implemented in the future).

  ### Operations

  Both types support the following:
  - Construction from signed and unsigned 32-bit and 64 integers,
  float/double or raw binary representation
  - Conversions between each other, to float or double and to int
  using @ref cvRound, @ref cvTrunc, @ref cvFloor, @ref cvCeil or a bunch of
  saturate_cast functions
  - Add, subtract, multiply, divide, remainder, square root, FMA with absolute precision
  - Comparison operations
  - Explicit sign, exponent and significand manipulation through get/set methods,
 number state indicators (isInf, isNan, isSubnormal)
  - Type-specific constants like eps, minimum/maximum value, best pi approximation, etc.
  - min(), max(), abs(), exp(), log() and pow() functions

*/

```

```c++
//! @{

```

## 238. Sample 38985

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/optim.hpp

Comment Line Number: 80

Label: empty

```c++
/** @brief Getter for the optimized function.

    The optimized function is represented by Function interface, which requires derivatives to
    implement the calc(double*) and getDim() methods to evaluate the function.

    @return Smart-pointer to an object that implements Function interface - it represents the
    function that is being optimized. It can be empty, if no function was given so far.
     */

```

```c++
    
```

## 239. Sample 39103

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/opengl.hpp

Comment Line Number: 218

Label: empty

```c++
/** @brief Unbind any buffers from the specified binding point.

    @param target Binding point. See cv::ogl::Buffer::Target .
     */

```

```c++
    
```

## 240. Sample 39110

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/opengl.hpp

Comment Line Number: 282

Label: class

```c++
/** @brief Smart pointer for OpenGL 2D texture memory with reference counting.
 */

```

```c++
class CV_EXPORTS Texture2D
{

```

## 241. Sample 39254

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/ocl.hpp

Comment Line Number: 794

Label: empty

```c++
/** Indicates if the image format is supported.
    */

```

```c++
    
```

## 242. Sample 39313

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/cuda.hpp

Comment Line Number: 294

Label: empty

```c++
//! returns element type

```

```c++
    
```

## 243. Sample 39361

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/cuda.hpp

Comment Line Number: 701

Label: other

```c++
/**< Event uses blocking synchronization */

```

```c++
        BLOCKING_SYNC  = 0x01,  
```

## 244. Sample 39447

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/affine.hpp

Comment Line Number: 42

Label: NOTHING

```c++
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

```

```c++

```

## 245. Sample 39584

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/mat.hpp

Comment Line Number: 1449

Label: empty

```c++
/** @brief Allocates new array data if needed.

    This is one of the key Mat methods. Most new-style OpenCV functions and methods that produce arrays
    call this method for each output array. The method uses the following algorithm:

    -# If the current array shape and the type match the new ones, return immediately. Otherwise,
       de-reference the previous data by calling Mat::release.
    -# Initialize the new header.
    -# Allocate the new data of total()\*elemSize() bytes.
    -# Allocate the new, associated with the data, reference counter and set it to 1.

    Such a scheme makes the memory management robust and efficient at the same time and helps avoid
    extra typing for you. This means that usually there is no need to explicitly allocate output arrays.
    That is, instead of writing:
    @code
        Mat color;
        ...
        Mat gray(color.rows, color.cols, color.depth());
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    you can simply write:
    @code
        Mat color;
        ...
        Mat gray;
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    because cvtColor, as well as the most of OpenCV functions, calls Mat::create() for the output array
    internally.
    @param rows New number of rows.
    @param cols New number of columns.
    @param type New matrix type.
     */

```

```c++
    
```

## 246. Sample 39675

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/mat.hpp

Comment Line Number: 2199

Label: func_decl

```c++
//! copy constructor

```

```c++
    Mat_(const Mat& m);
    
```

## 247. Sample 39823

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/mat.hpp

Comment Line Number: 2783

Label: call

```c++
//! returns the size of i-th matrix dimension (or 0)

```

```c++
    const int* size() const;
    
```

## 248. Sample 39948

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/mat.hpp

Comment Line Number: 3144

Label: func_decl

```c++
//! constructor that sets the iterator to the specified element of the matrix

```

```c++
    MatIterator_(Mat_<_Tp>* _m, int _row, int _col=0);
    
```

## 249. Sample 39978

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/mat.hpp

Comment Line Number: 3241

Label: func_decl

```c++
//! the copy constructor

```

```c++
    SparseMatIterator(SparseMat* _m, const int* idx);
    
```

## 250. Sample 40040

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/private.hpp

Comment Line Number: 106

Label: empty

```c++
// Returns a static string if there is a parallel framework,

```

```c++
    
```

## 251. Sample 40058

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/private.hpp

Comment Line Number: 192

Label: macro

```c++
// Different results

```

```c++
#define IPP_DISABLE_REMAP               1 
```

## 252. Sample 40501

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/persistence.hpp

Comment Line Number: 321

Label: other

```c++
//!< flag, write rawdata in Base64 by default. (consider using WRITE_BASE64)

```

```c++
        BASE64      = 64,     
```

## 253. Sample 40507

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/persistence.hpp

Comment Line Number: 367

Label: empty

```c++
/** @brief Checks whether the file is opened.

     @returns true if the object is associated with the current file and false otherwise. It is a
     good practice to call this method after you tried to open a file.
     */

```

```c++
    
```

## 254. Sample 40676

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/types_c.h

Comment Line Number: 158

Label: other

```c++
/**< bad number of channels, for example, some functions accept only single channel matrices */

```

```c++
 CV_BadNumChannels=            -15,  
```

## 255. Sample 40727

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/types_c.h

Comment Line Number: 329

Label: var_def

```c++
/**< Most of OpenCV functions support 1,2,3 or 4 channels */

```

```c++
    int  nChannels;         
```

## 256. Sample 41366

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/core_c.h

Comment Line Number: 1766

Label: func_def

```c++
/** Returns a set element by index. If the element doesn't belong to the set,
   NULL is returned */

```

```c++
CV_INLINE CvSetElem* cvGetSetElem( const CvSet* set_header, int idx )
{

```

## 257. Sample 41510

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/core_c.h

Comment Line Number: 2862

Label: other

```c++
/*!
 STL-style Sequence Iterator inherited from the CvSeqReader structure
*/

```

```c++
template<typename _Tp> class SeqIterator : public CvSeqReader
{

```

## 258. Sample 41624

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/imgproc/include/opencv2/imgproc/imgproc_c.h

Comment Line Number: 429

Label: func_decl

```c++
/** @brief Substitutes the last retrieved contour with the new one

   (if the substitutor is null, the last retrieved contour is removed from the tree)
@see cvFindContours
*/

```

```c++
CVAPI(void)   cvSubstituteContour( CvContourScanner scanner, CvSeq* new_contour );

```

## 259. Sample 42001

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/include/opencv2/gapi/gkernel.hpp

Comment Line Number: 256

Label: empty

```c++
// Prework: model "Device" API before it gets to G-API headers.

```

```c++
    
```

## 260. Sample 42133

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/include/opencv2/gapi/garray.hpp

Comment Line Number: 72

Label: call

```c++
// Default constructor

```

```c++
    protected:
        GArrayU();                                
```

## 261. Sample 42207

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/include/opencv2/gapi/gtype_traits.hpp

Comment Line Number: 95

Label: empty

```c++
//   for internal storage.

```

```c++
    
```

## 262. Sample 42233

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/include/opencv2/gapi/gtype_traits.hpp

Comment Line Number: 151

Label: empty

```c++
// namespace cv

```

```c++
} 
```

## 263. Sample 42603

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/backends/fluid/gfluidbackend.cpp

Comment Line Number: 1317

Label: other

```c++
// FIXME:

```

```c++
);
    
```

## 264. Sample 43068

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/backends/fluid/gfluidimgproc.cpp

Comment Line Number: 1540

Label: empty

```c++
//     DST     SRC     OP              __VA_ARGS__

```

```c++
        
```

## 265. Sample 43208

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/backends/fluid/gfluidimgproc_func.simd.hpp

Comment Line Number: 700

Label: var_def

```c++
// previous

```

```c++
                v_uint16 t0 = vx_load_expand(&in[k][l - shift]);  
```

## 266. Sample 43248

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/backends/fluid/gfluidimgproc_func.simd.hpp

Comment Line Number: 1143

Label: NOTHING

```c++
//-----------------------------
//
// Fluid kernels: Erode, Dilate
//
//-----------------------------

```

```c++

```

## 267. Sample 43577

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/compiler/passes/exec.cpp

Comment Line Number: 410

Label: empty

```c++
// _: Understand the contents and I/O connections of a new merged Island

```

```c++
        
```

## 268. Sample 43610

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/compiler/passes/kernels.cpp

Comment Line Number: 11

Label: macro

```c++
// util::indexed

```

```c++
#include <ade/util/zip_range.hpp>   
```

## 269. Sample 43613

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/src/compiler/passes/kernels.cpp

Comment Line Number: 33

Label: empty

```c++
// Generaly the algorithm is following

```

```c++
    
```

## 270. Sample 43823

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/java/generator/src/cpp/Mat.cpp

Comment Line Number: 1848

Label: NOTHING

```c++
//
//  Mat Mat::operator()(Range[] ranges)
//

```

```c++

```

## 271. Sample 44229

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/dnn/src/vkcom/src/op_relu.cpp

Comment Line Number: 74

Label: empty

```c++
// namespace cv::dnn::vkcom

```

```c++
}}} 
```

## 272. Sample 44567

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_perf.hpp

Comment Line Number: 719

Label: empty

```c++
//namespace comparators

```

```c++
} 
```

## 273. Sample 44725

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 977

Label: macro

```c++
// !defined(GTEST_HAS_HASH_MAP_)

```

```c++
#endif  
```

## 274. Sample 44739

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 1047

Label: macro

```c++
// To avoid conditional compilation we make it gtest-port.h's responsibility
// to #include the header implementing tuple.

```

```c++
#if GTEST_HAS_STD_TUPLE_
# include <tuple>  
```

## 275. Sample 44838

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 2303

Label: macro

```c++
// MS C++ compiler emits warning when a conditional expression is compile time
// constant. In some contexts this warning is false positive and needs to be
// suppressed. Use the following two macros in such cases:
//
// GTEST_INTENTIONAL_CONST_COND_PUSH_()
// while (true) {
// GTEST_INTENTIONAL_CONST_COND_POP_()
// }

```

```c++
# define GTEST_INTENTIONAL_CONST_COND_PUSH_() \
    GTEST_DISABLE_MSC_WARNINGS_PUSH_(4127)
# define GTEST_INTENTIONAL_CONST_COND_POP_() \
    GTEST_DISABLE_MSC_WARNINGS_POP_()

```

## 276. Sample 44880

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 2599

Label: empty

```c++
// references from r-values.

```

```c++
  
```

## 277. Sample 45219

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 4465

Label: empty

```c++
// The returned string is created using the ANSI codepage (CP_ACP) to

```

```c++
  
```

## 278. Sample 45261

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 4531

Label: call

```c++
// Not meant to be instantiated.

```

```c++
 private:
  String();  
```

## 279. Sample 45467

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 8355

Label: empty

```c++
// unsigned number x + N.

```

```c++
  
```

## 280. Sample 45718

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 10237

Label: NOTHING

```c++
// GOOGLETEST_CM0001 DO NOT DELETE

```

```c++

```

## 281. Sample 46065

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 12550

Label: other

```c++
// No implementation - assignment is unsupported.

```

```c++
 private:
  
```

## 282. Sample 46101

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 15174

Label: other

```c++
// No implementation - assignment is unsupported.

```

```c++
 private:
  
```

## 283. Sample 46359

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 19608

Label: macro

```c++
// Silence C4100 (unreferenced formal parameter) and 4805
// unsafe mix of type 'const int' and type 'const bool'

```

```c++
#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4805)
# pragma warning(disable:4100)
#endif

```

## 284. Sample 46549

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 20523

Label: empty

```c++
// Increments the death test count, returning the new count.

```

```c++
  
```

## 285. Sample 46912

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/ts/include/opencv2/ts/ts_gtest.h

Comment Line Number: 21683

Label: empty

```c++
// The current parameter value. Is also available in the test fixture's

```

```c++
  
```

## 286. Sample 47011

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/hal/intrin_neon.hpp

Comment Line Number: 826

Label: macro

```c++
// ARMv8, which adds support for 64-bit floating-point (so CV_SIMD128_64F is defined),

```

```c++
#if CV_SIMD128_64F
    
```

## 287. Sample 47115

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/hal/intrin_vsx.hpp

Comment Line Number: 23

Label: NOTHING

```c++
//! @cond IGNORED

```

```c++

```

## 288. Sample 47297

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/hal/intrin_cpp.hpp

Comment Line Number: 1522

Label: call

```c++
/** @brief Store data to memory (higher half)

Store higher half of register contents to memory.
Scheme:
@code
  REG {A B C D} ==> MEM {C D}
@endcode */

```

```c++
template<typename _Tp, int n>
inline void v_store_high(_Tp* ptr, const v_reg<_Tp, n>& a)
{

```

## 289. Sample 47746

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/cuda/simd_functions.hpp

Comment Line Number: 142

Label: call

```c++
// extract low halfword

```

```c++
    #if __CUDA_ARCH__ >= 300
        asm("vabsdiff2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vabsdiff.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vabsdiff.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s, t, u, v;
        s = a & 0x0000ffff; 
```

## 290. Sample 47755

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/cuda/simd_functions.hpp

Comment Line Number: 151

Label: other

```c++
// minimum of both halfwords

```

```c++
        s = v | s;          
```

## 291. Sample 47981

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/cuda/scan.hpp

Comment Line Number: 53

Label: NOTHING

```c++
/** @file
 * @deprecated Use @ref cudev instead.
 */

```

```c++

```

## 292. Sample 48190

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/opencl/runtime/autogenerated/opencl_clamdblas_impl.hpp

Comment Line Number: 387

Label: NOTHING

```c++
//openclamdblas_fn18(OPENCLAMDBLAS_FN_clAmdBlasCher2, clAmdBlasStatus, (clAmdBlasOrder p1, clAmdBlasUplo p2, size_t p3, cl_float2 p4, const cl_mem p5, size_t p6, int p7, const cl_mem p8, size_t p9, int p10, cl_mem p11, size_t p12, size_t p13, cl_uint p14, cl_command_queue* p15, cl_uint p16, const cl_event* p17, cl_event* p18))
//clAmdBlasStatus (*clAmdBlasCher2)(clAmdBlasOrder, clAmdBlasUplo, size_t, cl_float2, const cl_mem, size_t, int, const cl_mem, size_t, int, cl_mem, size_t, size_t, cl_uint, cl_command_queue*, cl_uint, const cl_event*, cl_event*) =
//        OPENCLAMDBLAS_FN_clAmdBlasCher2_switch_fn;
//static const struct DynamicFnEntry clAmdBlasCher2_definition = { "clAmdBlasCher2", (void**)&clAmdBlasCher2};

```

```c++

```

## 293. Sample 48207

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/opencl/runtime/autogenerated/opencl_clamdblas_impl.hpp

Comment Line Number: 472

Label: NOTHING

```c++
//openclamdblas_fn17(OPENCLAMDBLAS_FN_clAmdBlasCtbsv, clAmdBlasStatus, (clAmdBlasOrder p1, clAmdBlasUplo p2, clAmdBlasTranspose p3, clAmdBlasDiag p4, size_t p5, size_t p6, const cl_mem p7, size_t p8, size_t p9, cl_mem p10, size_t p11, int p12, cl_uint p13, cl_command_queue* p14, cl_uint p15, const cl_event* p16, cl_event* p17))
//clAmdBlasStatus (*clAmdBlasCtbsv)(clAmdBlasOrder, clAmdBlasUplo, clAmdBlasTranspose, clAmdBlasDiag, size_t, size_t, const cl_mem, size_t, size_t, cl_mem, size_t, int, cl_uint, cl_command_queue*, cl_uint, const cl_event*, cl_event*) =
//        OPENCLAMDBLAS_FN_clAmdBlasCtbsv_switch_fn;
//static const struct DynamicFnEntry clAmdBlasCtbsv_definition = { "clAmdBlasCtbsv", (void**)&clAmdBlasCtbsv};

```

```c++

```

## 294. Sample 48345

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/opencl/runtime/autogenerated/opencl_clamdblas_impl.hpp

Comment Line Number: 1181

Label: other

```c++
/*&clAmdBlasAddScratchImage_definition*/
```

```c++
    NULL
```

## 295. Sample 48434

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/src/opencl/runtime/autogenerated/opencl_clamdblas_impl.hpp

Comment Line Number: 1272

Label: other

```c++
/*&clAmdBlasSdot_definition*/
```

```c++
,
    NULL
```

## 296. Sample 49044

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/gapi/include/opencv2/gapi/own/mat.hpp

Comment Line Number: 18

Label: macro

```c++
//std::shared_ptr

```

```c++
#include <memory>                   
```

## 297. Sample 49229

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/include/opencv2/videoio/legacy/constants_c.h

Comment Line Number: 60

Label: other

```c++
// OpenCV Image Sequence (e.g. img_%02d.jpg)

```

```c++
    CV_CAP_IMAGES = 2000,    
```

## 298. Sample 49365

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/videoio/include/opencv2/videoio/legacy/constants_c.h

Comment Line Number: 272

Label: other

```c++
// The alpha channel of RGB32 output image format.

```

```c++
    CV_CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA                = 529, 
```

## 299. Sample 49481

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/cuda/detail/color_detail.hpp

Comment Line Number: 106

Label: empty

```c++
//to YUV

```

```c++
        
```

## 300. Sample 49525

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_gl.hpp

Comment Line Number: 53

Label: macro

```c++
// OPENCV_CORE_OCL_RUNTIME_OPENCL_GL_HPP

```

```c++
#endif 
```

## 301. Sample 49656

File Path: /Users/apple/Documents/work1/cpp_repos/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_clamdblas.hpp

Comment Line Number: 254

Label: comment

```c++
//#define clAmdBlasCtbmv clAmdBlasCtbmv_pfn

```

```c++
//#define clAmdBlasCtbsv clAmdBlasCtbsv_pfn

```

