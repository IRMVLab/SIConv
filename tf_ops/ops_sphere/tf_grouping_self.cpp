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

REGISTER_OP("QueryAndInterpolationSphere")
	.Attr("layers:int")
	.Attr("relative_xyz: bool")//////////////////////////////////
	.Input("xyz1: float32")//(b,n,3)
	.Input("feature:float32")//(b,n,c)
	.Input("xyz2: float32")//(b,n,3)
	.Input("feature_density:float32")//(b,n,1)
	.Input("lh: float32")//(b)
	.Output("weight_space:float32")//(b,n,27,3)
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * n * c
		c->WithRank(c->input(1), 3, &dims1);
		::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
		c->WithRank(c->input(2), 3, &dims2);
		int layers;
        	TF_RETURN_IF_ERROR(c->GetAttr("layers", &layers));
        if(layers==1){
		    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({ c->Dim(dims2, 0), c->Dim(dims2, 1), 15, c->Dim(dims1, 2) });
		    c->set_output(0, output);
		}
		else if(layers==2){
		    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({ c->Dim(dims2, 0), c->Dim(dims2, 1), 65, c->Dim(dims1, 2) });
		    c->set_output(0, output);
		}
		else if(layers==3){
		    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({ c->Dim(dims2, 0), c->Dim(dims2, 1), 175, c->Dim(dims1, 2) });
		    c->set_output(0, output);
		}
		return Status::OK();
	});
REGISTER_OP("QueryAndInterpolationSphereGrad")
	.Attr("layers:int")
	.Input("xyz1: float32")//(b,n,3)
	.Input("feature:float32")//(b.n.c)
	.Input("xyz2: float32")//(b,n,3)
	.Input("feature_density:float32")
	.Input("lh: float32")//(b)
	//.Input("relative_xyz: bool")////////////////
	.Input("grad_out:float32")//(b.n,[size],c)
	.Output("grad_points:float32")//(b,n,c)
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(1));
		return Status::OK();
	});

////////////////////////////////////////////////
void queryAndInterpolationSphereLauncher(int b, int n, int m, int c, int layers, const float *xyz1, const float *feature, const float *xyz2,const float *feature_density, const float *lh, bool relative_xyz, float *weight_space);
class QueryAndInterpolationSphereGpuOp : public OpKernel {
public:
	explicit QueryAndInterpolationSphereGpuOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("layers", &layers_));
	        OP_REQUIRES(context, layers_ > 0, errors::InvalidArgument("QueryBallPoint expects positive layers"));
        OP_REQUIRES_OK(context, context->GetAttr("relative_xyz", &relative_xyz_));
	}

	void Compute(OpKernelContext* context) override {

		const Tensor& xyz1_tensor = context->input(0);
		OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
		int b = xyz1_tensor.shape().dim_size(0);
		int n = xyz1_tensor.shape().dim_size(1);

		const Tensor& feature_tensor = context->input(1);
		OP_REQUIRES(context, feature_tensor.dims() == 3 && feature_tensor.shape().dim_size(1)==xyz1_tensor.shape().dim_size(1), errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, c) feature shape."));
		int c = feature_tensor.shape().dim_size(2);

		const Tensor& xyz2_tensor = context->input(2);
		OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
		int m = xyz2_tensor.shape().dim_size(1);


                const Tensor& feature_density_tensor = context->input(3);
                OP_REQUIRES(context, feature_density_tensor.dims() == 3 && feature_density_tensor.shape().dim_size(1) ==xyz1_tensor.shape().dim_size(1), errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 1) feature_density shape."));

		const Tensor& lh_tensor = context->input(4);
		OP_REQUIRES(context, lh_tensor.dims() == 1,errors::InvalidArgument("QueryBallPoint expects (b) lh shape"));

		//bool relative_xyz;
		/////////////////////////////////////////////////////////////
		//const Tensor& relative_xyz_tensor = context->input(5);
		//OP_REQUIRES(context, relative_xyz_tensor.dims() == 1,errors::InvalidArgument("QueryBallPoint expects (1) relative_xyz shape"));

		Tensor *weight_space_tensor = nullptr;
		if(layers_==1) OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,15,c}, &weight_space_tensor));
		else if(layers_==2) OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,65,c}, &weight_space_tensor));
		else if(layers_==3) OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,175,c}, &weight_space_tensor));

		auto xyz1_flat = xyz1_tensor.flat<float>();
		const float *xyz1 = &(xyz1_flat(0));
		auto feature_flat = feature_tensor.flat<float>();
		const float *feature = &(feature_flat(0));
		auto xyz2_flat = xyz2_tensor.flat<float>();
		const float *xyz2 = &(xyz2_flat(0));
                auto feature_density_flat = feature_density_tensor.flat<float>();
                const float *feature_density = &(feature_density_flat(0));
		auto lh_flat = lh_tensor.flat<float>();
		const float *lh = &(lh_flat(0));
		////////////////////////////////////////
		//auto relative_xyz_flat = relative_xyz_tensor.flat<bool>();
		//const bool *relative_xyz = &(relative_xyz_flat(0));
		///////////////////////////////////////////////////////
		auto weight_space_flat = weight_space_tensor->flat<float>();
		float *weight_space = &(weight_space_flat(0));
		queryAndInterpolationSphereLauncher(b, n, m, c, layers_, xyz1, feature, xyz2,feature_density, lh, relative_xyz_, weight_space);
	}
	private:
        	int layers_;
        	bool relative_xyz_;////////////////////////////
};
REGISTER_KERNEL_BUILDER(Name("QueryAndInterpolationSphere").Device(DEVICE_GPU), QueryAndInterpolationSphereGpuOp);

//////////////////////////////////////
void queryAndInterpolationSphereGradLauncher(int b, int n, int m, int c, int layers, const float *xyz1, const float *feature, const float *xyz2,const float *feature_density, const float *lh, const float *grad_out, float *grad_points);
class QueryAndInterpolationSphereGradGpuOp : public OpKernel {
public:
	explicit QueryAndInterpolationSphereGradGpuOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("layers", &layers_));
                OP_REQUIRES(context, layers_ > 0, errors::InvalidArgument("QueryBallPoint expects positive layers"));
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& xyz1_tensor = context->input(0);
		OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
		int b = xyz1_tensor.shape().dim_size(0);
		int n = xyz1_tensor.shape().dim_size(1);

		const Tensor& feature_tensor = context->input(1);
		OP_REQUIRES(context,  feature_tensor.dims() == 3 && feature_tensor.shape().dim_size(1)==xyz1_tensor.shape().dim_size(1), errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, c) feature shape."));
		int c = feature_tensor.shape().dim_size(2);

		const Tensor& xyz2_tensor = context->input(2);
		OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
		int m = xyz2_tensor.shape().dim_size(1);


                const Tensor& feature_density_tensor = context->input(3);
                OP_REQUIRES(context, feature_density_tensor.dims() == 3 && feature_density_tensor.shape().dim_size(1) ==xyz1_tensor.shape().dim_size(1), errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 1) feature_density shape."));
		const Tensor& lh_tensor = context->input(4);
		OP_REQUIRES(context, lh_tensor.dims() == 1,errors::InvalidArgument("QueryBallPoint expects (b) lh shape"));

		//////////////////////////////////////////////////////////////
		//const Tensor& relative_xyz_tensor = context->input(5);
		//OP_REQUIRES(context, relative_xyz_tensor.dims() == 1,errors::InvalidArgument("QueryBallPoint expects (1) relative_xyz shape"));

		const Tensor& grad_out_tensor = context->input(5);

		Tensor *grad_points_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{ b,n,c }, &grad_points_tensor));

		auto xyz1_flat = xyz1_tensor.flat<float>();
		const float *xyz1 = &(xyz1_flat(0));
		auto feature_flat = feature_tensor.flat<float>();
		const float *feature = &(feature_flat(0));
		auto xyz2_flat = xyz2_tensor.flat<float>();
		const float *xyz2 = &(xyz2_flat(0));
		auto feature_density_flat = feature_density_tensor.flat<float>();
                const float *feature_density = &(feature_density_flat(0));
		auto lh_flat = lh_tensor.flat<float>();
		const float *lh = &(lh_flat(0));
		//////////////////////////////////
		//auto relative_xyz_flat = relative_xyz_tensor.flat<bool>();
		//const bool *relative_xyz = &(relativez_xyz_flat(0));
		////////////////////////////
		auto grad_out_flat = grad_out_tensor.flat<float>();
		const float *grad_out = &(grad_out_flat(0));
		auto grad_points_flat = grad_points_tensor->flat<float>();
		float *grad_points = &(grad_points_flat(0));
		///////////////////////////////////////////////////
		queryAndInterpolationSphereGradLauncher(b, n, m, c, layers_, xyz1, feature, xyz2,feature_density, lh, grad_out, grad_points);
	}
	private:
	    int layers_;
};
REGISTER_KERNEL_BUILDER(Name("QueryAndInterpolationSphereGrad").Device(DEVICE_GPU), QueryAndInterpolationSphereGradGpuOp);
