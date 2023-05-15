# VL-Counter
Zero-shot object counting model with Vison-Language Model


### Requirements
~~~bash
    pip install git+https://github.com/openai/CLIP.git
~~~

### Multiclass Samples
- 336.jpg: blueberries, strawberries
- 343.jpg: kiwis, strawberries


## Experiments

**1. Add CrossEntorpy Loss to density map**
- 물체가 있는 부분을 0으로 예측했을 때 패널티가 큼  
  → 전반적인 값이 높게 예측됨