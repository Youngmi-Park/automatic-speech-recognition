# Automatic Speech Recognition(ASR)

기존 ASR에서는 P(W|O)를 모델링하기 위해 은닉 마르코프 모델-가우시안 혼합 모델(HMM-GMM)을 사용했다. 
하지만 딥러닝의 발달로, 은닉 마르코프 모델 - 심층 신경망 모델(HMM-DNN)과 딥러닝 기반의 종단간 모델(End-to-End)이 등장했고 
실제 두 모델들은 HMM-GMM 보다 더 좋은 성능을 보여주었다.
<br><br>
## HMM based ASR
음향(Acoustic), Lexicon, 언어(Language) 모델 3가지 부분으로 나누어지며. 각 부분들은 서로 독립적으로 모델링된다.


HMM 기반의 방법은 음성 인식 과정이 3가지로 나누어져 있기 때문에 나타나는 단점들이 있다. 
* 각각의 모델들에서 최적의 성능을 만들었다 하더라도 이것이 ASR 모델 전체에서 최적의 성능(Global Optima)을 내지 못할 수도 있다.
* HMM 계산의 간결성을 위해 조건부 독립을 가정하였는데 이로 인해 실제 계산과의 차이가 생긴다.
* 음향(Acoustic), Lexicon, 언어(Language) 모델을 각각 구축하는 과정에서 많은 전문 지식이 요구된다.

하지만 다른 도메인에 쉽게 적용시킬 수 있다는 장점도 있다. <br>
e.g. 기존 언어 모델을 뉴스 코퍼스에서 학습시킨 언어 모델로 바꿔 모델을 뉴스 음성인식에 사용할 수 있음.
<br><br>
## Open Source
* ~~Kaldi~~<br>
레시피 미지원<br>
<빌드를 성공 후 사용을 위해 가이드를 살펴 보았으나, Unix-like 시스템에서만 할 수 있고,<br>
미리 구성된 레시피를 이용해야 하는데 윈도 네이티브로는 사용이 불가능하다.

* ~~Julius~~<br>
2014년 이후 업데이트 중단 상태

* DeepSpeech2  (테스트 중)
  + Paepr Review
  + [AI Hub](http://www.aihub.or.kr/aidata/105) 한국어 음성 데이터 신청 및 다운로드
  + Language Model 공부
  + code Review 
  
  

<br><br>
## Mozilla Deep Speech
https://github.com/mozilla/DeepSpeech

### About
중국 ‘바이두(baidu)’에서 공개한 End-to-End 음성 인식 모델 DeepSpeech 연구 논문을 기반으로 한 오픈소스 음성 인식 엔진 <br>
(소스 코드가 모태는 아니고, 기계학습 모델 논문을 참고한 것) <br>
다양한 기능과 사용 편의성을 고려할 때 최고의 음성 인식 도구 중 하나이다. **character-based CTC를 사용하는 모델**로 Mozilla Common Voice 데이터 셋에 대해 학습 되었으며, Mozilla Public License에 따라 라이선스가 부여된다. 기계학습 엔진으로는 TensorFlow를 활용하였고, 엔진 코드는 C/C++로 구현되어 있다. 때문에 C/C++은 완벽히 연동되고, nodejs, python은 패키지 매니저를 통해 이용 가능하다. GO, Rust 같은 언어 지원도 별도 프로젝트로 진행하고 있다고 한다. 사용자 지정 데이터셋을 사용하여 학습 가능하다. 

<br>
가장 큰 장점은 모델 파일을 다운로드하고 몇 분 내에 로컬에서 추론을 수행할 수 있다는 것이다.
하지만 DeepSpeech는 다른 언어에 활용하기는 어려움이 있는데, 이는 예측하려는 언어로 사용자 지정 데이터셋을 사용해 모델을 미세 조정하면 된다.
엔진의 아키텍처는 원래 Deep Speech : Scaling up end-to-end 에서 제시된 것에서 동기가 부여되었다. 그러나 현재 엔진은 원래의 엔진과 많은 측면에서 다르다. 엔진의 핵심은 음성 스펙트로 그램을 수집하고 영어 텍스트 필사본을 생성하도록 훈련 된 RNN(recurrent neural network)이다.<br>


### Using a Pre-trained Model
https://github.com/Youngmi-Park/automatic-speech-recognition/wiki/Using-a-Pretrained-Model

### Paper Review
1. [Deep Speech: Scaling up end-to-end speech recognition, Awni H., Carl C., Jared C., Bryan](https://github.com/Youngmi-Park/automatic-speech-recognition/edit/main/paper%20review.md)
2. [Mozilla deepspeech](https://github.com/Youngmi-Park/automatic-speech-recognition/blob/main/mozilla-deepspeech.md)

## Reference

[1] Deep Speech: Scaling up end-to-end speech recognition, Awni H., Carl C., Jared C., Bryan C.
https://arxiv.org/abs/1412.5567

[2] https://ichi.pro/ko/mozilla-deepspeechleul-sayonghayeo-jadong-eulo-jamag-saengseong-94748643243687

[3] https://ratsgo.github.io/speechbook/docs/neuralam/deepspeech
