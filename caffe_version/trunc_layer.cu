#include <algorithm>
#include <vector>

#include "caffe/layers/trunc_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TruncForward(const int n, const Dtype* in, Dtype* out,
    Dtype scale) {
  CUDA_KERNEL_LOOP(index, n) {
	 out[index] =in[index] > scale ? scale : (in[index] < (-1*scale) ? (-1*scale) : in[index])  ;
/*if(in[index] >= -scale && in[index] <= scale){
          out[index] =in[index];
        }else if(in[index] < -scale) {
            out[index] = -scale;
        }else if(in[index] > scale) {
            out[index] = scale;
      }*/
  }
}

template <typename Dtype>
void TruncLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype scale = this->layer_param_.trunc_param().scale();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TruncForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, scale);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void TruncBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype scale) {
  CUDA_KERNEL_LOOP(index, n) {		
		out_diff[index] = in_diff[index] *((in_data[index] > (-1*scale) && in_data[index]<scale) );
 /* if(in_data[index] > scale ||in_data[index] < -1*scale){
     out_diff[index] = 0;
  }else {
      out_diff[index] = in_diff[index];
  }*/
}
}

  

template <typename Dtype>
void TruncLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
Dtype scale = this->layer_param_.trunc_param().scale();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TruncBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, scale);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(TruncLayer);


}  // namespace caffe
