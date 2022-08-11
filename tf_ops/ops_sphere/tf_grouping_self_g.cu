// input: L (1), xyz1 (b,n,3),xyz1_feature(b,n,c), xyz2 (b,m,3)
// output: weight_space (b,m,[size],c)
//#include <stdio.h>
////////////////////////////////////////////////
__global__ void query_and_interpolation_sphere_gpu(int b, int n, int m, int c, int layers, const float *xyz1, const float *feature, const float *xyz2, const float *feature_density,const float *lh, bool relative_xyz, float *weight_space) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    feature += n*c*batch_index;
    xyz2 += m*3*batch_index;
    feature_density += n*batch_index;
    lh += batch_index;
    int size_;
    if(layers==1) {size_=15;}
    else if(layers==2) {size_=65;}
    else if(layers==3) {size_=175;}
    weight_space += m*size_*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    const int BufferSize=512;
    float l=lh[0];
    float alpha1=l*l;
    float alpha2=l*l*0.2345*0.2345;
    float alpha3=l*l*0.1696*0.1696;
    float alpha=4*4*l*l;

    //int layer_points[7]={1,3,7,12,7,3,1};
    int layer1_points[5]={1,3,7,3,1};
    int layer2_points[9]={1,3,7,12,19,12,7,3,1};
    int layer3_points[13]={1,3,7,12,19,27,37,27,19,12,7,3,1};

    //float layer[68]={0.0, 0.0, 0.5, 0.2887, -0.5, 0.2887, 0.0, 0.5774, 0.0, 0.0, 1.0, 0.0, 0.5, 0.8660, -0.5, 0.8660, -1.0, 0.0, -0.5, -0.8660, 0.5, -0.8660,
    //                0.0, 0.5774, -0.5, -0.2887, 0.5, -0.2887, 1.0, 0.5774, 0.5, 1.4434, -0.5, 1.4434, -1.0, 0.5774, -1.5, -0.2887, -1.0, -1.1547, 0.0, -1.1547, 1.0, -1.1547, 1.5, -0.2887,
    //                0.0, 0.0, 1.0, 0.0, 0.5, 0.8660, -0.5, 0.8660, -1.0, 0.0, -0.5, -0.8660, 0.5, -0.8660,0.5, 0.2887, -0.5, 0.2887, 0.0, 0.5774, 0.0, 0.0};

    float layers1[30]={0.0,0.0, 0.5,0.2887, -0.5,0.2887, 0.0,-0.5774, 0.0,0.0, 1.0,0.0, 0.5,0.8660, -0.5,0.8660, -1.0,0.0, -0.5,-0.8660, 0.5,-0.8660, 0.5,0.2887, -0.5,0.2887, 0.0,-0.5774, 0.0,0.0};
    float layers2[130]={0.0,0.0, 0.25,0.1443, -0.25,0.1443, 0.0,0.2887, 0.0,0.0, 0.5,0.0, 0.25,0.4330, -0.25,0.4330, -0.5,0.0, -0.25,-0.4330, 0.25,-0.4330, 0.25,0.1443, -0.25,0.1443, 0.0,0.2887,
					0.75,0.1443, 0.5, 0.5774, 0.0,0.5774, -0.5,0.5774, -0.75,0.1443, -0.5,-0.2887, -0.25,-0.7217, 0.25,-0.7217, 0.5,-0.2887, 0.0,0.0, 0.5,0.0, 0.25,0.4330, -0.25,0.4330, -0.25,-0.4330,
					0.25,-0.4330, 0.5,0.0, 1.0,0.0, 0.75,0.4330, 0.5,0.8660, 0.0,0.8660, -0.5,0.8660, -0.75,0.4330, -1.0,0.0, -0.75,-0.4330, -0.5,-0.8660, 0.0,-0.8660, 0.5,-0.8660, 0.75,-0.4330,
					0.25,0.1443, -0.25,0.1443, 0.0,0.2887, 0.75,0.1443, 0.5, 0.5774, 0.0,0.5774, -0.5,0.5774, -0.75,0.1443, -0.5,-0.2887, -0.25,-0.7217, 0.25,-0.7217, 0.5,-0.2887, 0.0,0.0, 0.5,0.0,
					0.25,0.4330, -0.25,0.4330, -0.5,0.0, -0.25,-0.4330, 0.25,-0.4330, 0.5,0.2887, -0.5,0.2887, 0.0,-0.5774, 0.0,0.0};
    float layers3[350]={0.0,0.0, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887, -0.3333,0.0, -0.1667,-0.2887, 0.1667,-0.2887,
					0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849, 0.0,0.3849, -0.3333,0.3849, -0.5,0.0962, -0.3333,-0.1925, -0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925,
					0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887, -0.1667,-0.2887, 0.1667,-0.2887, 0.3333,0.0, 0.6667,0.0, 0.5,0.2887, 0.3333,0.5773, 0.0,0.5773, -0.3333,0.5773, -0.5,0.2887,
					-0.6667,0.0, -0.5,-0.2887, -0.3333,-0.5773, 0.0,-0.5773, 0.3333,-0.5773, 0.5,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849, 0.0,0.3849, -0.3333,0.3849,
					-0.5,0.0962, -0.3333,-0.1925, -0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925, 0.8333,0.0962, 0.6667,0.3849, 0.5,0.6736, 0.1667,0.6736, -0.1667,0.6736, -0.5,0.6736, -0.6667,0.3849,
					-0.8333,0.0962, -0.6667,-0.1925, -0.5,-0.4811, -0.3333,-0.7698, 0.0,-0.7698, 0.3333,-0.7698, 0.5,-0.4811, 0.6667,-0.1925, 0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887,
					-0.1667,-0.2887, 0.1667,-0.2887, 0.3333,0.0, 0.6667,0.0, 0.5,0.2887, 0.3333,0.5773, 0.0,0.5773, -0.3333,0.5773, -0.5,0.2887, -0.6667,0.0, -0.5,-0.2887, -0.3333,-0.5773, 0.0,-0.5773,
					0.3333,-0.5773, 0.5,-0.2887, 1.0,0.0, 0.8333,0.2887, 0.6667,0.5774, 0.5, 0.8660, 0.1667,0.8660, -0.1667,0.8660, -0.5,0.8660, -0.6667,0.5774, -0.8333,0.2887, -1.0,0.0, -0.8333,-0.2887,
					-0.6667,-0.5774, -0.5,-0.8660, -0.1667,-0.8660, 0.1667,-0.8660, 0.5,-0.8660, 0.6667,-0.5774, 0.8333,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849,
					0.0,0.3849, -0.3333,0.3849, -0.5,0.0962, -0.3333,-0.1925, -0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925, 0.8333,0.0962, 0.6667,0.3849, 0.5,0.6736, 0.1667,0.6736, -0.1667,0.6736,
					-0.5,0.6736, -0.6667,0.3849, -0.8333,0.0962, -0.6667,-0.1925, -0.5,-0.4811, -0.3333,-0.7698, 0.0,-0.7698, 0.3333,-0.7698, 0.5,-0.4811, 0.6667,-0.1925, 0.0,0.0, 0.3333,0.0,
					0.1667,0.2887, -0.1667,0.2887, -0.1667,-0.2887, 0.1667,-0.2887, 0.3333,0.0, 0.6667,0.0, 0.5,0.2887, 0.3333,0.5773, 0.0,0.5773, -0.3333,0.5773, -0.5,0.2887, -0.6667,0.0, -0.5,-0.2887,
					-0.3333,-0.5773, 0.0,-0.5773, 0.3333,-0.5773, 0.5,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849, 0.0,0.3849, -0.3333,0.3849, -0.5,0.0962, -0.3333,-0.1925,
					-0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925, 0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887, -0.3333,0.0, -0.1667,-0.2887, 0.1667,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.0,0.0};

    for(int j=index;j<m;j+=stride){
	float dist_sum[175];
	int pts=0;
	bool flag=true;
	for(int f=0;f<175;++f) dist_sum[f]=0;
	for(int a=0;a<size_;++a){
	    for(int d=0;d<c;++d){
		weight_space[j*size_*c+a*c+d]=0;
	    }
	}
	for(int k=0;k<n;++k){
	    float x2=xyz2[j*3+0];
	    float y2=xyz2[j*3+1];
	    float z2=xyz2[j*3+2];
		float x1 = xyz1[k * 3 + 0];
        float y1 = xyz1[k * 3 + 1];
        float z1 = xyz1[k * 3 + 2];
	    float d=max(((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
	    if (d<=alpha) {
     		int cnt = 0;
     		    if(layers==1){
     		        for(int o=-2;o<=2 && flag;++o){
     		            for(int p=0;p<layer1_points[o+2] && flag;++p){
     		                float x2_=x2+layers1[cnt*2+0]*l*4/2.4495;
     		                float y2_=y2+layers1[cnt*2+1]*l*4/2.4495;
     		                float z2_=z2+o*l*4.0/3.0;
     		                float dist = max(((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1)),1e-20f);
			                if(dist<=alpha1){
			                    dist_sum[cnt] += 1.0/ (sqrtf(dist)*feature_density[k]);
			                    if(relative_xyz==true){
								for(int g=0;g<3;++g){
			   	                    weight_space[j * size_ * c + cnt * c + g] += (feature[k*c+g]-xyz2[j*3+g]) /  (sqrtf(dist)*feature_density[k]);
			   	                }
			   	                for(int g=3;g<c;++g){
			   	                    weight_space[j * size_ * c + cnt * c + g] += feature[k*c+g] /  (sqrtf(dist)*feature_density[k]);
			   	                }
								}
								else{
			   	                for(int g=0;g<c;++g){
			   	                    weight_space[j * size_ * c + cnt * c + g] += feature[k*c+g] /  (sqrtf(dist)*feature_density[k]);
			   	                }
								}
			   	                pts+=1;
				                if(pts>=BufferSize) flag=false;
			                }
			                cnt+=1;
			            }
     		        }
     		        if(!flag) break;
     		    }
     		    if(layers==2){
     		        for(int o=-4;o<=4 && flag;++o){
     		            for(int p=0;p<layer2_points[o+4] && flag;++p){
     		                float x2_=x2+layers2[cnt*2+0]*l;
     		                float y2_=y2+layers2[cnt*2+1]*l;
     		                float z2_=z2+o*l*0.3125;
     		                float dist = max(((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1)),1e-20f);
			                if(dist<=alpha2){
			                    dist_sum[cnt] += 1/ (sqrtf(dist)*feature_density[k]);
			                        for(int g=0;g<3;++g){
			   	                        weight_space[j * size_ * c + cnt * c + g] += (feature[k*c+g]-xyz2[j*3+g]) /  (sqrtf(dist)*feature_density[k]);
			   	                    }
			   	                    for(int g=3;g<c;++g){
			   	                        weight_space[j * size_ * c + cnt * c + g] += feature[k*c+g] /  (sqrtf(dist)*feature_density[k]);
			   	                    }
			   	                pts+=1;
				                if(pts>=BufferSize) flag=false;
			                }
			                cnt+=1;
			            }
     		        }
     		        if(!flag) break;
     		    }
     		    if(layers==3){
     		        for(int o=-6;o<=6 && flag;++o){
     		            for(int p=0;p<layer3_points[o+6] && flag;++p){
     		                float x2_=x2+layers3[cnt*2+0]*l;
     		                float y2_=y2+layers3[cnt*2+1]*l;
     		                float z2_=z2+o*l*0.2260;
     		                float dist = max(((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1)),1e-20f);
			                if(dist<=alpha3){
			                    dist_sum[cnt] += 1/ (sqrtf(dist)*feature_density[k]);
			                        for(int g=0;g<3;++g){
			   	                        weight_space[j * size_ * c + cnt * c + g] += (feature[k*c+g]-xyz2[j*3+g]) /  (sqrtf(dist)*feature_density[k]);
			   	                    }
			   	                    for(int g=3;g<c;++g){
			   	                        weight_space[j * size_ * c + cnt * c + g] += feature[k*c+g] /  (sqrtf(dist)*feature_density[k]);
			   	                    }
			   	                pts+=1;
				                if(pts>=BufferSize) flag=false;
			                }
			                cnt+=1;
			            }
     		        }
     		        if(!flag) break;
     		    }
	        }
	    }
	for (int s = 0; s < size_; ++s) {
	    for(int h=0;h<c;++h){
		if(dist_sum[s]!=0) {
		    weight_space[j * size_ * c + s * c + h] *= (1.0/dist_sum[s]);
		}
	    }
	}
    }
}
//xyz1(b,n,3),feature(b,n,c),xyz2(b,m,3),grad_out(b,m,27,c),grad_points(b,n,c)
////////////////////////////////////////
__global__ void query_and_interpolation_sphere_grad_gpu(int b, int n, int m, int c, int layers, const float *xyz1,const float *feature, const float *xyz2,const float *feature_density, const float *lh, const float *grad_out, float *grad_points) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    feature += n*c*batch_index;
    xyz2 += m*3*batch_index;
    feature_density += n*batch_index;
    lh += batch_index;
    int size_;
    if(layers==1) {size_=15;}
    else if(layers==2) {size_=65;}
    else if(layers==3) {size_=175;}
    grad_points += n*c*batch_index;
    grad_out += m*size_*c*batch_index;

    //int layer_points[7]={1,3,7,12,7,3,1};
    int layer1_points[5]={1,3,7,3,1};
    int layer2_points[9]={1,3,7,12,19,12,7,3,1};
    int layer3_points[13]={1,3,7,12,19,27,37,27,19,12,7,3,1};


    //float layer[68]={0.0, 0.0, 0.5, 0.2887, -0.5, 0.2887, 0.0, 0.5774, 0.0, 0.0, 1.0, 0.0, 0.5, 0.8660, -0.5, 0.8660, -1.0, 0.0, -0.5, -0.8660, 0.5, -0.8660,
    //                0.0, 0.5774, -0.5, -0.2887, 0.5, -0.2887, 1.0, 0.5774, 0.5, 1.4434, -0.5, 1.4434, -1.0, 0.5774, -1.5, -0.2887, -1.0, -1.1547, 0.0, -1.1547, 1.0, -1.1547, 1.5, -0.2887,
    //                0.0, 0.0, 1.0, 0.0, 0.5, 0.8660, -0.5, 0.8660, -1.0, 0.0, -0.5, -0.8660, 0.5, -0.8660,0.5, 0.2887, -0.5, 0.2887, 0.0, 0.5774, 0.0, 0.0};

    float layers1[30]={0.0,0.0, 0.5,0.2887, -0.5,0.2887, 0.0,-0.5774, 0.0,0.0, 1.0,0.0, 0.5,0.8660, -0.5,0.8660, -1.0,0.0, -0.5,-0.8660, 0.5,-0.8660, 0.5,0.2887, -0.5,0.2887, 0.0,-0.5774, 0.0,0.0};
    float layers2[130]={0.0,0.0, 0.25,0.1443, -0.25,0.1443, 0.0,0.2887, 0.0,0.0, 0.5,0.0, 0.25,0.4330, -0.25,0.4330, -0.5,0.0, -0.25,-0.4330, 0.25,-0.4330, 0.25,0.1443, -0.25,0.1443, 0.0,0.2887,
					0.75,0.1443, 0.5, 0.5774, 0.0,0.5774, -0.5,0.5774, -0.75,0.1443, -0.5,-0.2887, -0.25,-0.7217, 0.25,-0.7217, 0.5,-0.2887, 0.0,0.0, 0.5,0.0, 0.25,0.4330, -0.25,0.4330, -0.25,-0.4330,
					0.25,-0.4330, 0.5,0.0, 1.0,0.0, 0.75,0.4330, 0.5,0.8660, 0.0,0.8660, -0.5,0.8660, -0.75,0.4330, -1.0,0.0, -0.75,-0.4330, -0.5,-0.8660, 0.0,-0.8660, 0.5,-0.8660, 0.75,-0.4330,
					0.25,0.1443, -0.25,0.1443, 0.0,0.2887, 0.75,0.1443, 0.5, 0.5774, 0.0,0.5774, -0.5,0.5774, -0.75,0.1443, -0.5,-0.2887, -0.25,-0.7217, 0.25,-0.7217, 0.5,-0.2887, 0.0,0.0, 0.5,0.0,
					0.25,0.4330, -0.25,0.4330, -0.5,0.0, -0.25,-0.4330, 0.25,-0.4330, 0.5,0.2887, -0.5,0.2887, 0.0,-0.5774, 0.0,0.0};
    float layers3[350]={0.0,0.0, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887, -0.3333,0.0, -0.1667,-0.2887, 0.1667,-0.2887,
					0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849, 0.0,0.3849, -0.3333,0.3849, -0.5,0.0962, -0.3333,-0.1925, -0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925,
					0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887, -0.1667,-0.2887, 0.1667,-0.2887, 0.3333,0.0, 0.6667,0.0, 0.5,0.2887, 0.3333,0.5773, 0.0,0.5773, -0.3333,0.5773, -0.5,0.2887,
					-0.6667,0.0, -0.5,-0.2887, -0.3333,-0.5773, 0.0,-0.5773, 0.3333,-0.5773, 0.5,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849, 0.0,0.3849, -0.3333,0.3849,
					-0.5,0.0962, -0.3333,-0.1925, -0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925, 0.8333,0.0962, 0.6667,0.3849, 0.5,0.6736, 0.1667,0.6736, -0.1667,0.6736, -0.5,0.6736, -0.6667,0.3849,
					-0.8333,0.0962, -0.6667,-0.1925, -0.5,-0.4811, -0.3333,-0.7698, 0.0,-0.7698, 0.3333,-0.7698, 0.5,-0.4811, 0.6667,-0.1925, 0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887,
					-0.1667,-0.2887, 0.1667,-0.2887, 0.3333,0.0, 0.6667,0.0, 0.5,0.2887, 0.3333,0.5773, 0.0,0.5773, -0.3333,0.5773, -0.5,0.2887, -0.6667,0.0, -0.5,-0.2887, -0.3333,-0.5773, 0.0,-0.5773,
					0.3333,-0.5773, 0.5,-0.2887, 1.0,0.0, 0.8333,0.2887, 0.6667,0.5774, 0.5, 0.8660, 0.1667,0.8660, -0.1667,0.8660, -0.5,0.8660, -0.6667,0.5774, -0.8333,0.2887, -1.0,0.0, -0.8333,-0.2887,
					-0.6667,-0.5774, -0.5,-0.8660, -0.1667,-0.8660, 0.1667,-0.8660, 0.5,-0.8660, 0.6667,-0.5774, 0.8333,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849,
					0.0,0.3849, -0.3333,0.3849, -0.5,0.0962, -0.3333,-0.1925, -0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925, 0.8333,0.0962, 0.6667,0.3849, 0.5,0.6736, 0.1667,0.6736, -0.1667,0.6736,
					-0.5,0.6736, -0.6667,0.3849, -0.8333,0.0962, -0.6667,-0.1925, -0.5,-0.4811, -0.3333,-0.7698, 0.0,-0.7698, 0.3333,-0.7698, 0.5,-0.4811, 0.6667,-0.1925, 0.0,0.0, 0.3333,0.0,
					0.1667,0.2887, -0.1667,0.2887, -0.1667,-0.2887, 0.1667,-0.2887, 0.3333,0.0, 0.6667,0.0, 0.5,0.2887, 0.3333,0.5773, 0.0,0.5773, -0.3333,0.5773, -0.5,0.2887, -0.6667,0.0, -0.5,-0.2887,
					-0.3333,-0.5773, 0.0,-0.5773, 0.3333,-0.5773, 0.5,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.5,0.0962, 0.3333,0.3849, 0.0,0.3849, -0.3333,0.3849, -0.5,0.0962, -0.3333,-0.1925,
					-0.1667,-0.4811, 0.1667,-0.4811, 0.3333,-0.1925, 0.0,0.0, 0.3333,0.0, 0.1667,0.2887, -0.1667,0.2887, -0.3333,0.0, -0.1667,-0.2887, 0.1667,-0.2887, 0.1667,0.0962, -0.1667,0.0962, 0.0,0.1925, 0.0,0.0};

    const int BufferSize=512;
    int index = threadIdx.x;
    int stride = blockDim.x;
    float l=lh[0];
    float alpha1=l*l;
    float alpha2=l*l*0.2345*0.2345;
    float alpha3=l*l*0.1696*0.1696;
    float alpha=4*4*l*l;


    //const int Points=4096;
    //__shared__ float buf1[Points*3];

    for(int x=0;x<n;++x){
	for(int y=0;y<c;++y){
	    grad_points[x*c+y]=0;
	}
    }
    /*for(int j=threadIdx.x;j<min(Points,n)*3;j+=blockDim.x){
      buf1[j]=xyz1[j];
    }
    __syncthreads();*/
    for (int j=index;j<m;j+=stride) {
	float dist_sum[175];
	float buf[BufferSize];
	int idx[BufferSize*2];
	int pts=0;
	bool flag=true;
	for(int f=0;f<175;++f){
	    dist_sum[f]=0;
	}
	for (int k = 0; k < n; ++k) {
	    float x2 = xyz2[j * 3 + 0];
	    float y2 = xyz2[j * 3 + 1];
	    float z2 = xyz2[j * 3 + 2];
	    float x1 = xyz1[k * 3 + 0];
        float y1 = xyz1[k * 3 + 1];
        float z1 = xyz1[k * 3 + 2];
	    //if(feature_density[k]==0) continue;
	    /*float x1,y1,z1;
	    if(k<Points){
	        x1 = buf1[k * 3 + 0];
	        y1 = buf1[k * 3 + 1];
	        z1 = buf1[k * 3 + 2];
	    }
	    else {
		    x1 = xyz1[k * 3 + 0];
            y1 = xyz1[k * 3 + 1];
            z1 = xyz1[k * 3 + 2];
	    }*/
	    float d=max(((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
	    if (d <=alpha) {
		    int cnt=0;
		        if(layers==1){
     		        for(int o=-2;o<=2 && flag;++o){
     		            for(int p=0;p<layer1_points[o+2] && flag;++p){
     		                float x2_=x2+layers1[cnt*2+0]*l*4/2.4495;
     		                float y2_=y2+layers1[cnt*2+1]*l*4/2.4495;
     		                float z2_=z2+o*l*4.0/3.0;
     		                float dist = max(((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1)),1e-20f);
			                if(dist<=alpha1){
			                    dist_sum[cnt] += 1.0/ (sqrtf(dist)*feature_density[k]);
			                    idx[pts*2+0]=k;
				                idx[pts*2+1]=cnt;
				                buf[pts]=1.0/ (sqrtf(dist)*feature_density[k]);
				                pts+=1;
				                if(pts>=BufferSize) flag=false;
			   	                }
			                cnt+=1;
     		                }
     		        }
     		        if(!flag) break;
     		    }
     		    if(layers==2){
     		        for(int o=-4;o<=4 && flag;++o){
     		            for(int p=0;p<layer2_points[o+4] && flag;++p){
     		                float x2_=x2+layers2[cnt*2+0]*l;
     		                float y2_=y2+layers2[cnt*2+1]*l;
     		                float z2_=z2+o*l*0.3125;
     		                float dist = max(((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1)),1e-20f);
			                if(dist<=alpha2){
			                    dist_sum[cnt] += 1/ (sqrtf(dist)*feature_density[k]);
			                    idx[pts*2+0]=k;
				                idx[pts*2+1]=cnt;
				                buf[pts]=1/ (sqrtf(dist)*feature_density[k]);
				                pts+=1;
				                if(pts>=BufferSize) flag=false;
			   	                }
			                cnt+=1;
     		                }
     		        }
     		        if(!flag) break;
     		    }
     		    if(layers==3){
     		        for(int o=-6;o<=6 && flag;++o){
     		            for(int p=0;p<layer3_points[o+6] && flag;++p){
     		                float x2_=x2+layers3[cnt*2+0]*l;
     		                float y2_=y2+layers3[cnt*2+1]*l;
     		                float z2_=z2+o*l*0.2260;
     		                float dist = max(((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1)),1e-20f);
			                if(dist<=alpha3){
			                    dist_sum[cnt] += 1/ (sqrtf(dist)*feature_density[k]);
			                    idx[pts*2+0]=k;
				                idx[pts*2+1]=cnt;
				                buf[pts]=1/ (sqrtf(dist)*feature_density[k]);
				                pts+=1;
				                if(pts>=BufferSize) flag=false;
			   	                }
			                cnt+=1;
     		                }
     		        }
     		        if(!flag) break;
     		    }
	    }
	}
	for(int h=0;h<pts;++h){
	    for(int g=0;g<c;++g){
		if(dist_sum[idx[h*2+1]]!=0) {
		atomicAdd(&grad_points[idx[h*2+0]*c+g],(grad_out[j*size_*c+idx[h*2+1]*c+g]*buf[h]/dist_sum[idx[h*2+1]]));
		}
	    }
	}
    }
}

///////////////////////////////////
void queryAndInterpolationSphereLauncher(int b, int n, int m, int c, int layers, const float *xyz1, const float *feature, const float *xyz2,const float *feature_density,const float *lh, bool relative_xyz, float *weight_space) {
    query_and_interpolation_sphere_gpu<<<b,256>>>(b,n,m,c,layers,xyz1,feature,xyz2,feature_density,lh,relative_xyz,weight_space);
    //cudaDeviceSynchronize();
}
void queryAndInterpolationSphereGradLauncher(int b, int n, int m, int c, int layers, const float *xyz1, const float *feature, const float *xyz2,const float *feature_density, const float *lh, const float *grad_out, float *grad_points) {
    query_and_interpolation_sphere_grad_gpu<<<b,256>>>(b,n,m,c,layers,xyz1,feature,xyz2,feature_density,lh,grad_out,grad_points);
    //cudaDeviceSynchronize();
}
