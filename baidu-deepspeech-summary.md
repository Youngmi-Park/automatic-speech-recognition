# Deep Speech: Scaling up end-to-end speech recognition

Awni Hannun∗ , Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen, Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, Andrew Y. Ng 
Baidu Research – Silicon Valley AI Lab


## Abstract

이 논문에서는 End-to-end deep learning을 사용하여 개발된 최첨단 음성 인식 시스템을 소개한다.

복잡한 처리 파이프라인에 의존하는 전통적인 음성 시스템보다 훨씬 간단한 구조를 가지고 있다.

또 기존 시스템은 시끄러운 환경에서 사용할 때 성능이 저하되는 경향이 있는데, 이 시스템은 배경의 소음, 잔향 또는 화자의 변화를 모델링하기 위해 hand-designed component를 사용하지 않고 대신 이러한 효과에 대응하는 기능을 직접 학습한다. **"음소"** 라는 개념도 없기 때문에 음소 사전이 필요하지 않다. 이 접근방식의 핵심은 최적화된 여러 GPU를 사용하는 RNN 훈련 시스템과 학습을 위해 많은 양의 다양한 데이터를 효율적으로 얻을 수 있는 일련의 새로운 데이터 합성 기술이다. Deep Speech는 많은 분야에서 연구된 Switchboard Hub5’00에서 이전에 발표된 결과를 능가하여 전체 테스트 셋에서 16.0 % 오류를 달성했습니다. 또한 널리 사용되는 최첨단 상용 음성 시스템보다 시끄러운 환경에서 음성을 더 잘 처리한다.

*end-to-end deep learning: 처음부터 끝까지라는 의미로 데이터(입력)에서 목표한 결과(출력)를 사람이 개입 없이 얻는다는 뜻을 담고 있다.

## 1. Introduction
최고의 음성 인식 시스템은 여러 알고리즘과 수작업 설계 처리 단계로 구성된 정교한 파이프 라인에 의존한다. 이 논문에서는 Deep Speech라는 end-to-end 음성 시스템을 소개한다. 이는 수작업의 처리단계를 딥러닝이 대체하는 것을 말한다. 언어 모델과 결합된 이 접근방식은 어려운 음성 인식 작업에서 기존 방법보다 더 놓은 성능을 보이며 훨씬 간단하다. 이러한 결과는 여러 GPU와 수천 시간의 데이터를 사용하여 대규모 RNN (Recurrent Neural Network)을 훈련함으로써 가능하다. 시스템은 데이터에서 직접 학습하므로 스피커 적응 또는 노이즈 필터링을 위한 특수 구성 요소가 필요하지 않다. (사실, 화자 변화 및 소음에 대한 견고성이 중요한 설정에서 이 시스템은 탁월하다.) Deep Speech는 Switchboard Hub5’00 corpus에서 이전에 게시된 방법보다 성능이 우수하며 16.0 %의 오류를 달성하였다. 또 소음이 있는 음성 인식 테스트에서 상용 시스템보다 우수한 성능을 발휘한다.


전통적인 음성 시스템은 특화된 입력 기능, 음향 모델 및 HMM (Hidden Markov Model)을 포함하여 고도의 처리 단계를 많이 사용한다. 이러한 파이프 라인을 개선하려면 전문가가 기능과 모델을 조정하는 데 많은 노력을 기울여야한다. 딥 러닝 알고리즘의 도입은 일반적으로 음향 모델을 개선하여 음성 시스템 성능을 향상시켰다. 이러한 개선은 중요하지만 딥러닝은 여전히 전통적인 음성 파이프 라인에서 제한된 역할을 한다. 결과적으로 시끄러운 환경에서 음성인식과 같은 성능을 향상시키려면 견고성을 위해 나머지 시스템 부분을 힘들게 설계해야 한다. 대조적으로 이 시스템은 RNN을 사용하여 end-to-end deep learning을 적용한다. 대규모 데이터 세트에서 학습하여 전반적인 성능을 향상시킨다. 이 모델은 transcription을 생성하기 위해 end-to end 학습을 했기 때문에 충분한 데이터와 컴퓨팅 성능을 통해 자체적으로 소음이나 화자 변형에 대한 견고성을 학습할 수 있다.

 

그러나 end-to-end deep learning의 이점을 활용하는데 몇 가지 문제가 있다.

1) 레이블된 대규모 훈련 세트를 구축하는 혁신적인 방법을 찾아야한다.
2) 모든 데이터를 효과적으로 활용하기에 충분히 큰 네트워크를 훈련할 수 있어야한다.

음성 시스템에서 레이블된 데이터를 처리하기위한 한 가지 과제는 입력 음성과 텍스트 대본의 정렬을 찾는 것인데 이러한 문제는 
[13] A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: ´ Labelling unsegmented sequence data with recurrent neural networks. In ICML, pages 369– 376. ACM, 20 
이 논문에서 다루었으며 신경망이 훈련 중에 정렬되지 않은 오디오를 쉽게 사용할 수 있다.

한편, 아래의 논문은 대규모 신경망에서의 빠른 훈련을 다루고 있는데, 다중 GPU 계산의 속도 이점을 보여준다.
[7] A. Coates, B. Huval, T. Wang, D. J. Wu, A. Y. Ng, and B. Catanzaro. Deep learning with COTS HPC. In International Conference on Machine Learning, 2013.
이 논문에서는 이러한 통찰력을 활용해 대규모 음성 데이터 세트 및 확장 가능한 RNN 학습을 기반으로 복잡한 기존 방법을 능가할 수 있는 일반 학습 시스템을 달성하는 것을 목표로 한다. 부분적으로 아래의 논문에서 영감을 받았으며 수작업을 대체하기 위해 초기 비지도 학습 기술을 적용했다.

[27] H. Lee, P. Pham, Y. Largman, and A. Y. Ng. Unsupervised feature learning for audio classification using convolutional deep belief networks. In Advances in Neural Information Processing Systems, pages 1096–1104, 2009
GPU에 잘 매핑하기 위해 특별히 RNN 모델을 선택했으며 병렬화를 개선하기 위해 새로운 모델 파티션 체계를 사용한다. 또한 시스템에서 왜곡된 대량의 레이블된 음성 데이터를 조립하는 프로세스를 제안한다. 수집 및 합성된 데이터의 조합을 사용하여 시스템은 사실적인 소음과 화자 변화에 대한 견고함을 학습한다. (Lombard effect 포함.) 

이러한 아이디어를 종합하면 기존 파이프 라인보다 한 번에 더 간단하면서도 어려운 음성 작업에서 더 나은 성능을 발휘하는 end-to-end speech system을 구축하는 데 충분하다. Deep Speech는 full Switchboard Hub5’00 테스트 셋에서 16.0%의 오류율을 달성한다. (게시된 것들 중 가장 좋은 결과) 또한 자체 구성한 시끄러운 환경에서의 음성 인식 데이터 셋에서 19.1%의 word error rate를 달성했다. (최고의 상용 시스템이 30.5 %의 word error rate)


## 2. RNN Training Setup – 모델과 훈련 프레임워크에 대한 설명
![image](https://user-images.githubusercontent.com/53163222/108289963-c0ffea00-71d2-11eb-81c9-5bd531e951cb.png)

### 2.1 Regularization

RNN의 분산을 더 줄이기 위해 몇 가지 기술을 사용한다. 훈련 중에 드롭 아웃[19]을 5 %-10 % 사이로 설정한다. Feed forward 레이어에는 드롭 아웃을 적용하지만 recurrent hidden activations에는 적용하지 않는다. (네트워크 평가 중 컴퓨터 비전에서 일반적으로 사용되는 기술은 변환 또는 반사에 의해 입력을 무작위로 지터하고, 결과를 투표하거나 평균하는 것이다.)[23] 지터링은 음성인식에서 일반적인 방법은 아니지만 원본 오디오 파일을 왼쪽과 오른쪽으로 5ms만큼 변환한 다음 재계산 후 순방향 전파하고 출력 확률의 평균을 구하는 것이 좋다는 것을 알았다.

테스트에서는 여러 RNN의 앙상블을 사용하여 동일한 방식으로 출력의 평균을 구한다.

*** Dropout**은 인공 신경망의 각 레이어 노드에서 학습할 때 마다 일부 노드를 사용하지 않고 학습을 진행한다. 최종적으로 인공 신경망으로도 과적합을 방지해주며 실제 테스트에서도 좋은 성능을 보여준다.

[19] G. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. abs/1406.7806, 2014. http://arxiv.org/abs/1406.7806.

[23] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1106–1114, 2012.

### 2.2 Language Model

대량의 레이블된 음성 데이터에서 학습하면 RNN 모델은 읽을 수 있는 문자 수준의 transcription을 생성하는 방법을 배울 수 있다. (실제로 RNN에 의해 예측되는 가장 가능성 있는 문자열은 거의 정확하다. – 외부 언어 제약없이) RNN에 의해 만들어진 오류는 영어 단어의 음성학적으로 그럴듯한 표현인 경향이 있으며 피하기 어렵다.

![image](https://user-images.githubusercontent.com/53163222/108289967-c4937100-71d2-11eb-8403-5c62dafd8bb3.png)

모든 단어나 언어 구조를 듣기 위해 충분한 음성 데이터로 훈련하는 것은 비현실적이기 때문에 이 시스템은 N-gram 언어 모델과 통합하여 레이블이 없는 거대한 텍스트 말뭉치에서 쉽게 학습 할 수 있도록 한다. 비교를 위해 우리의 음성 데이터 세트에는 일반적으로 최대 3 백만 개의 발화가 포함되지만 섹션 5.2의 실험에 사용된 N-gram 언어 모델은 2 억 2 천만 구문의 말뭉치에서 훈련되어 495,000 단어의 어휘를 지원한다.

RNN의 출력이 주어지면 문자의 시퀀스를 찾기 위해 검색을 수행한다. c1, c2 …는 RNN 출력과 언어 모델 (언어 모델이 문자열을 단어로 해석)에 따라 가장 가능성이 높다. 특히 결합된 Q(c)를 최대화하는 시퀀스 c를 찾는 것을 목표로 한다. 

<img src="https://user-images.githubusercontent.com/53163222/108290171-2e137f80-71d3-11eb-8070-b1b3bbf37db4.png">: RNN의 출력
<img src="https://user-images.githubusercontent.com/53163222/108290175-3075d980-71d3-11eb-93d8-8b87e9940e23.png">: 문자 시퀀스

![image](https://user-images.githubusercontent.com/53163222/108290178-3370ca00-71d3-11eb-8fd0-0b63dfeb921f.png)<br>
여기서 α와 β는 RNN, 언어 모델 제약 조건 및 문장 길이 사이의 균형을 제어하는 조정 가능한 매개 변수 (교차 검증으로 설정)이다.

<img src="(https://user-images.githubusercontent.com/53163222/108290182-35d32400-71d3-11eb-8828-bc2266ebe9b5.png">:  N-gram 모델에 따른 시퀀스 c의 확률

Hannun 등이 설명한 접근 방식과 유사한 1000-8000 범위의 일반적인 빔 크기로 고도로 최적화된 빔 검색 알고리즘을 사용하여 목표를 최대화한다.[16].

[16] A. Y. Hannun, A. L. Maas, D. Jurafsky, and A. Y. Ng. First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs. abs/1408.2873, 2014. http://arxiv.org/abs/1408.2873.

 

## 3. Optimizations - gpu 최적화

이 논문에서는 네트워크를 고속 실행 (따라서 빠른 교육)할 수 있도록 몇 가지 설계를 했다. 예를 들어 구현이 간단하고 고도로 최적화된 몇 가지 BLAS 호출에만 의존하는 homogeneous 정류 선형 네트워크를 선택했다. 중요한. 실험 속도를 높이기 위해 다중 GPU 학습 [7, 23]을 사용하지만 이를 효과적으로 수행하려면 몇 가지 추가 작업이 필요하다.
<img src="">

<img src="">
