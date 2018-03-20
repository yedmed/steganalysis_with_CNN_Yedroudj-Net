#include <algorithm>
#include <vector>

#include "caffe/layers/trunc_layer.hpp"

namespace caffe {

template <typename Dtype>
void TruncLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype scale = this->layer_param_.trunc_param().scale();
  for (int i = 0; i < count; ++i) {
//    top_data[i] = std::max(bottom_data[i], Dtype(0))
//        + negative_slope * std::min(bottom_data[i], Dtype(0));
        top_data[i] = bottom_data[i] > scale ? scale :(bottom_data[i] < (-1*scale) ? (-1*scale): bottom_data[i]) ;
/*      if(bottom_data[i] >= -scale && bottom_data[i] <= scale){
          top_data[i] =bottom_data[i];
        }else if(bottom_data[i] < -scale) {
            top_data[i] = -scale;

        }else if(bottom_data[i] > scale) {
            top_data[i] = scale;
      }*/
  }
}

template <typename Dtype>
void TruncLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
      Dtype scale = this->layer_param_.trunc_param().scale();
    for (int i = 0; i < count; ++i) {
      //bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)+ negative_slope * (bottom_data[i] <= 0));
		  		
    bottom_diff[i] = top_diff[i] *((bottom_data[i] > (-1*scale) && bottom_data[i]<scale) );
       /* if(bottom_data[i] > scale || bottom_data[i] < -scale){
            bottom_diff[i] =0;
          }else {
              bottom_diff[i] = top_diff[i];
          }*/

    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(TruncLayer);
#endif

INSTANTIATE_CLASS(TruncLayer);

}  // namespace caffe
