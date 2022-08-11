// input: xyz (b,n,3),feature(b,n,c), radius(b)
// output: feature_density(b,n,1)
__global__ void find_feature_density_gpu(int b, int n, int c, const float *xyz, const float *feature, const float *radius, float *feature_density) {
    int batch_index = blockIdx.x;
    xyz += n*3*batch_index;
    feature += n*batch_index;
    radius += batch_index;
	feature_density += n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    float r=radius[0]*radius[0];

    for(int j=index;j<n;j+=stride){
		feature_density[j] = 0.0;
		float x1=xyz[j*3+0];
		float y1=xyz[j*3+1];
		float z1=xyz[j*3+2];
		for(int k=0;k<n;++k){
			float x2=xyz[k*3+0];
			float y2=xyz[k*3+1];
			float z2=xyz[k*3+2];
			float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
			if (d<=r) {
				feature_density[j] += feature[k];
			}
		}
	}
}

/*
__global__ void find_feature_density_grad_gpu(int b, int n, int c, const float *xyz, const float *feature, const float *radius, const float *grad_out, float *grad_points) {
    int batch_index = blockIdx.x;
    xyz += n*3*batch_index;
    feature += n*c*batch_index;
    radius += batch_index;
	grad_out += n*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    float r=radius[0]*radius[0];

    for(int j=index;j<n;j+=stride){
        for(int h=0;h<c;++h){
		    grad_points[j*c+h] = 0.0;
        }
    }
    for(int j=index;j<n;j+=stride){
		float feat1_norm = 0.0;
		for(int g=0;g<c;++g){
			feat1_norm += feature[j*c+g]*feature[j*c+g];
		}
		feat1_norm = sqrtf(feat1_norm);
		float x1=xyz[j*3+0];
		float y1=xyz[j*3+1];
		float z1=xyz[j*3+2];
		for(int k=0;k<n;++k){
			float x2=xyz[k*3+0];
			float y2=xyz[k*3+1];
			float z2=xyz[k*3+2];
			float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
			if (d<=r) {
				float feat2_norm = 0;
				for(int p=0;p<c;++p){
				    feat2_norm += feature[k*c+p]*feature[k*c+p];
				}
				feat2_norm = sqrtf(feat2_norm);
				float feat1_feat2 = 0.0;
     			for(int h=0;h<c;++h){
					feat1_feat2 += feature[j*c+h]*feature[k*c+h];
				}
				float dist_relative = max(1.0-sqrtf(d/r),0.0);
				for(int h=0;h<c;++h){
				    float grad1 = dist_relative*(feature[k*c+h]*feat1_norm-feat1_feat2*feature[j*c+h]/feat1_norm)/(feat1_norm*feat1_norm*feat2_norm);
				    //float grad2 = dist_relative*(feature[j*c+h]*feat2_norm-feat1_feat2*feature[k*c+h]/feat2_norm)/(feat2_norm*feat2_norm*feat1_norm);
				    atomicAdd(&grad_points[j*c+h], grad1*grad_out[j]);
				    //atomicAdd(&grad_points[k*c+h], grad2*grad_out[k]);
				    //grad_points[j*c+h] += grad1*grad_out[j];
				    //grad_points[k*c+h] += grad2*grad_out[k];
				}
			}
		}
	}
}
*/

void findFeatureDensityLauncher(int b, int n, int c, const float *xyz, const float *feature, const float *radius, float *feature_density) {
    find_feature_density_gpu<<<b,256>>>(b,n,c,xyz,feature,radius,feature_density);
    //cudaDeviceSynchronize();
}
/*
void findFeatureDensityGradLauncher(int b, int n, int c, const float *xyz, const float *feature, const float *radius, const float *grad_out, float *grad_points) {
    find_feature_density_grad_gpu<<<b,256>>>(b,n,c,xyz,feature,radius,grad_out,grad_points);
    //cudaDeviceSynchronize();
}
*/