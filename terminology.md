# 용어 정리
### 스펙트로그램(Spectrogram) 
소리나 파동을 시각화하여 파악하기 위한 도구로, 파형과 스펙트럼의 특징이 조합되어 있다. 
x축은 시간(Time), y축은 주파수(Frequency), z 축은 진폭(Amplitude)을 나타낸다.

<figure class="imageblock alignCenter" data-filename="스펙트로그램.png" data-origin-width="644" data-origin-height="335"><span data-url="https://blog.kakaocdn.net/dn/6iL9D/btqDxvwgJXW/e9MIeg6zSoMm28ro3tpGCK/img.png" data-lightbox="lightbox" data-alt="[그림4] 스펙트로그램"><img src="https://blog.kakaocdn.net/dn/6iL9D/btqDxvwgJXW/e9MIeg6zSoMm28ro3tpGCK/img.png" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F6iL9D%2FbtqDxvwgJXW%2Fe9MIeg6zSoMm28ro3tpGCK%2Fimg.png" data-filename="스펙트로그램.png" data-origin-width="644" data-origin-height="335"></span><figcaption>

- 파형 (Waveform): 파형에서 <b>x축은 시간(Time), y축은 진폭(Amplitude)</b>을 나타낸다.
- 스펙트럼 (Spectrum): 스펙트럼에서 <b>x축은 주파수(Frequency) y축은 진폭(Amplitude)</b>을 나타낸다.

### Dropout
인공 신경망의 각 레이어 노드에서 학습할 때 마다 일부 노드를 사용하지 않고 학습을 진행한다.<br>
최종적으로 인공 신경망으로도 과적합을 방지해주며 실제 테스트에서도 좋은 성능을 보여준다.
: http://dmqm.korea.ac.kr/activity/seminar/273
