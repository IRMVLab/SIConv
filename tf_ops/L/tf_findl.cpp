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

REGISTER_OP("Findl")
	.Input("xyz: float32")//(b,n,3)
	.Output("l:float32")//(b)
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * n * 3
		c->WithRank(c->input(0), 3, &dims1);
		::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({ c->Dim(dims1, 0) });
		c->set_output(0, output);
		return Status::OK();
	});
	
void findl_cpu(int b, int n, const float *xyz, float *l){
	for (int i=0;i<b;++i) {
		float l_sum=0;
		for (int j=0;j<n;++j) {
			float x1=xyz[j*3+0];
			float y1=xyz[j*3+1];
			float z1=xyz[j*3+2];
			double dist_best=1e40;
			for(int k=0;k<n;++k){
				float x2=xyz[k*3+0];
				float y2=xyz[k*3+1];
				float z2=xyz[k*3+2];
				double dist=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
				if(dist<dist_best && dist!=0) dist_best=dist;
			}
			l_sum+=dist_best;
		}
		l[i]=l_sum/n;
		xyz+=n*3;
	}
}
class FindlOp : public OpKernel {
public:
	explicit FindlOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		
		const Tensor& xyz_tensor = context->input(0);
		OP_REQUIRES(context, xyz_tensor.dims() == 3 && xyz_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("Findl expects (batch_size, ndataset, 3) xyz shape."));
		int b = xyz_tensor.shape().dim_size(0);
		int n = xyz_tensor.shape().dim_size(1);
		
		Tensor *l_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b}, &l_tensor));

		auto xyz_flat = xyz_tensor.flat<float>();
		const float *xyz = &(xyz_flat(0));
		auto l_flat = l_tensor->flat<float>();
		float *l = &(l_flat(0));
		findl_cpu(b, n, xyz, l);
	}
};
REGISTER_KERNEL_BUILDER(Name("Findl").Device(DEVICE_CPU), FindlOp);
