## Automatic Speech Recognition(ASR)

기존 ASR에서는 P(W|O)를 모델링하기 위해 은닉 마르코프 모델-가우시안 혼합 모델(HMM-GMM)을 사용했다. 
하지만 딥러닝의 발달로, 은닉 마르코프 모델 - 심층 신경망 모델(HMM-DNN)과 딥러닝 기반의 종단간 모델(End-to-End)이 등장했고 
실제 두 모델들은 HMM-GMM 보다 더 좋은 성능을 보여주었다.

### HMM based ASR
음향(Acoustic), Lexicon, 언어(Language) 모델 3가지 부분으로 나누어지며. 각 부분들은 서로 독립적으로 모델링된다.


HMM 기반의 방법은 음성 인식 과정이 3가지로 나누어져 있기 때문에 나타나는 단점들이 있다. 
* 각각의 모델들에서 최적의 성능을 만들었다 하더라도 이것이 ASR 모델 전체에서 최적의 성능(Global Optima)을 내지 못할 수도 있다.
* HMM 계산의 간결성을 위해 조건부 독립을 가정하였는데 이로 인해 실제 계산과의 차이가 생긴다.
* 음향(Acoustic), Lexicon, 언어(Language) 모델을 각각 구축하는 과정에서 많은 전문 지식이 요구된다.

하지만 다른 도메인에 쉽게 적용시킬 수 있다는 장점도 있다. 
e.g. 기존 언어 모델을 뉴스 코퍼스에서 학습시킨 언어 모델로 바꿈으로써 전체 음성인식 모델을 뉴스 음성인식에 사용할 수 있음.


### Open Source
~~* Kaldi ~~<br>
레시피 미지원<br>
<빌드를 성공 후 사용을 위해 가이드를 살펴 보았으나, Unix-like 시스템에서만 할 수 있고,<br>
미리 구성된 레시피를 이용해야 하는데 윈도 네이티브로는 사용이 불가능하다.

~~* Julius~~<br>
2014년 이후 업데이트 중단 상태

* DeepSpeech2  (테스트하는 중)






Deep Speech
중국 대표 IT 기업 ‘바이두(baidu)’에서 공개한 End-to-End 음성 인식 모델 Deep Speech2 모델


Deep Speech: Scaling up end-to-end speech recognition, Awni H., Carl C., Jared C., Bryan C.
https://arxiv.org/abs/1412.5567
