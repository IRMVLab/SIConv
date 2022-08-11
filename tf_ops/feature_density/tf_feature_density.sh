CUDA_PATH=/usr/local/cuda-9.0
$CUDA_PATH/bin/nvcc tf_feature_density.cu -o tf_feature_density.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 tf_feature_density.cpp tf_feature_density.cu.o -o tf_feature_density_so.so -shared -fPIC -I $TF_INC -I $CUDA_PATH/include -L$TF_LIB -I$TF_INC/external/nsync/public -lcudart -L $CUDA_PATH/lib64/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
