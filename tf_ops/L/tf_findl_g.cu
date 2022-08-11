//xyz(b,n,3),l(b)
__global__ void findl_gpu(int b, int n, const float *xyz,float *l) {
    int batch_index = blockIdx.x;
    xyz += n*3*batch_index;
    l += batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int j=0;j<n-1;++1){
		float x1=xyz[j*3+0]
		float y1=xyz[j*3+1]
		float z1=xyz[j*3+2]
		float dist_best=1e20f;
		float l_sum=0;
		for(int k=j;k<n;++1){
			float x2=xyz[[j*3+0]
			float y2=xyz[j*3+1]
			float z2=xyz[j*3+2]
			float dist=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))
			if(dist<dist_best) dist_best=dist;
		}
		l_sum+=dist_best;
	}
	l[0]=l_sum/(n-1)
}

void findlLauncher(int b, int n, const float *xyz, float *l) {
    findl_gpu<<<b,256>>>(b,n,xyz,l);
    //cudaDeviceSynchronize();
}