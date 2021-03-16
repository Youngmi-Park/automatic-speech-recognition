||||음성과 관련된 데이터들은 오픈SRLlin|k 프로|젝트에서 체계적으로 제공을 하고 있다. 
h|ttp://www.openslr.org/resources|.p|hp||

LibriSpe|ech

Librispeech [Panayotov et al. 2015]는 현존 음성인식 연구에 있어서 가장 널리 사용되는 대규모 영어 음성 데이터 중 하나로 사용자 참여형 오디오북 프로젝트인 LibriVox projectlink의 결과물이다. Lirbrispeech는 16kHz로 샘플링된 약 1,000시간 분량의 녹음된 오디오북 데이터이다. LibriVox 데이터는 긴 발화 시간을 갖는 오디오북이기 때문에 음성인식 학습에 적합하도록 다양한 전처리를 수행하였다. 또한, 여러 가지 형태의 데이터 증강기법을 통해 다양한 버전의 코퍼스를 공개했다. 각 데이터 코퍼스의 구성은 아래 표와 같다.
|데이터 종류|시간(h)|화자당 시간(m)|여성 화자 수|남성 화자 수	|전체 화자 수
|Dev-clean	|5.4|	8	|20|	20|	40|
|Test-clean	|5.4	|8	|20	|20	|40|
|Dev-other	|5.3	|10	|16|	17	|33|
|Test-other	|5.1	|10	|17|	16	|33|
|Train-clean-100	|100.5|	25|	125	|126|	251|
|Train-clean-360	|363.6|	25|	439	|482	|921|


https://www.openslr.org/12