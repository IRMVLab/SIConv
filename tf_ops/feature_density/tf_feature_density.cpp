#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("FindFeatureDensity")
    .Input("xyz: float32")//(b,n,3)
    .Input("feature:float32")//(b,n,1)
    .Input("radius:float32")//(b)
    .Output("feature_density:float32")//(b,n,1)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	    ::tensorflow::shape_inference::ShapeHandle dims; // batch_size * n * 3
	    c->WithRank(c->input(0), 3, &dims);
	    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({ c->Dim(dims, 0), c->Dim(dims, 1), 1 });
	    c->set_output(0, output);
	    return Status::OK();
});
/*
REGISTER_OP("FindFeatureDensityGrad")
    .Input("xyz: float32")//(b,n,3)
    .Input("feature:float32")//(b,n,c)
    .Input("radius:float32")//(b)
    .Input("grad_out:float32")//(b,n,1)
    .Output("grad_points:float32")//(b,n,c)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
});
*/
void findFeatureDensityLauncher(int b, int n, int c, const float *xyz, const float *feature, const float *radius,float *feature_density);
class FindFeatureDensityGpuOp : public OpKernel {
public:
	explicit FindFeatureDensityGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {

		const Tensor& xyz_tensor = context->input(0);
		OP_REQUIRES(context, xyz_tensor.dims() == 3 && xyz_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz shape."));
		int b = xyz_tensor.shape().dim_size(0);
		int n = xyz_tensor.shape().dim_size(1);

		const Tensor& feature_tensor = context->input(1);
		OP_REQUIRES(context, feature_tensor.dims() == 3 && feature_tensor.shape().dim_size(1) == xyz_tensor.shape().dim_size(1), errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 1) feature shape."));
		int c = feature_tensor.shape().dim_size(2);

		const Tensor& radius_tensor = context->input(2);
		OP_REQUIRES(context, radius_tensor.dims() == 1, errors::InvalidArgument("QueryBallPoint expects (b) radius shape"));

		Tensor *feature_density_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{ b,n,1 }, &feature_density_tensor));

		auto xyz_flat = xyz_tensor.flat<float>();
		const float *xyz = &(xyz_flat(0));
		auto feature_flat = feature_tensor.flat<float>();
		const float *feature = &(feature_flat(0));
		auto radius_flat = radius_tensor.flat<float>();
		const float *radius = &(radius_flat(0));
		auto feature_density_flat = feature_density_tensor->flat<float>();
		float *feature_density = &(feature_density_flat(0));
		findFeatureDensityLauncher(b, n, c, xyz, feature, radius, feature_density);
	}
};
REGISTER_KERNEL_BUILDER(Name("FindFeatureDensity").Device(DEVICE_GPU), FindFeatureDensityGpuOp);
/*
void findFeatureDensityGradLauncher(int b, int n, int c, const float *xyz, const float *feature, const float *radius, const float *grad_out, float *grad_points);
class FindFeatureDensityGradGpuOp : public OpKernel {
public:
	explicit FindFeatureDensityGradGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {

		const Tensor& xyz_tensor = context->input(0);
		OP_REQUIRES(context, xyz_tensor.dims() == 3 && xyz_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz shape."));
		int b = xyz_tensor.shape().dim_size(0);
		int n = xyz_tensor.shape().dim_size(1);

		const Tensor& feature_tensor = context->input(1);
		OP_REQUIRES(context, feature_tensor.dims() == 3 && feature_tensor.shape().dim_size(1) == xyz_tensor.shape().dim_size(1), errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, c) feature shape."));
		int c = feature_tensor.shape().dim_size(2);

		const Tensor& radius_tensor = context->input(2);
		OP_REQUIRES(context, radius_tensor.dims() == 1, errors::InvalidArgument("QueryBallPoint expects (b) radius shape"));

        const Tensor& grad_out_tensor = context->input(3);

		Tensor *grad_points_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{ b,n,c }, &grad_points_tensor));

		auto xyz_flat = xyz_tensor.flat<float>();
		const float *xyz = &(xyz_flat(0));
		auto feature_flat = feature_tensor.flat<float>();
		const float *feature = &(feature_flat(0));
		auto radius_flat = radius_tensor.flat<float>();
		const float *radius = &(radius_flat(0));
		auto grad_out_flat = grad_out_tensor.flat<float>();
		const float *grad_out = &(grad_out_flat(0));
		auto grad_points_flat = grad_points_tensor->flat<float>();
		float *grad_points = &(grad_points_flat(0));
		findFeatureDensityGradLauncher(b, n, c, xyz, feature, radius, grad_out, grad_points);
	}
};
REGISTER_KERNEL_BUILDER(Name("FindFeatureDensityGrad").Device(DEVICE_GPU), FindFeatureDensityGradGpuOp);
*/
