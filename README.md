## Automatic Speech Recognition(ASR)

기존 ASR에서는 P(W|O)를 모델링하기 위해 은닉 마르코프 모델-가우시안 혼합 모델(HMM-GMM)을 사용했다. 
하지만 딥러닝의 발달로, 은닉 마르코프 모델 - 심층 신경망 모델(HMM-DNN)과 딥러닝 기반의 종단간 모델(End-to-End)이 등장했고 
실제 두 모델들은 HMM-GMM 보다 더 좋은 성능을 보여주었다.

### HMM based ASR
음향(Acoustic), Lexicon, 언어(Language) 모델 3가지 부분으로 나누어지며. 각 부분들은 서로 독립적으로 모델링된다.


오픈소스
-Kaldi
-Julius
-Wav2Letter++
-DeepSpeech2



Deep Speech
중국 대표 IT 기업 ‘바이두(baidu)’에서 공개한 End-to-End 음성 인식 모델 Deep Speech2 모델


<small>
Deep Speech: Scaling up end-to-end speech recognition
Awni Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen, Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, Andrew Y. Ng
https://arxiv.org/abs/1412.5567
  </small>
