# 🔎 Emochat
- Emotion Detecting ChatBot!

## ✏️시연 영상


https://github.com/singsingsing3/EmoChat-detecting-your-Emotion-with-cam-and-ChatBot-AI-/assets/120327265/91f3293e-1c77-4bd1-877e-badce24f9a56


## ✏️프로그램 설명
- 캠을 통해 당신의 표정을 읽어들이고 당신의 감정을 분석한 후에 당신의 감정을 고려하여 대화를 해주는 채팅 봇 프로그램입니다.



## ✏️사용 설명서
- 이 프로그램은 Chat GPT API를 이용하여 채팅모델을 구축했습니다. 따라서 개인의 API KEY가 필요합니다.
  => KEY를 발급받은 후에 ChatBot.py에서 당신의 OPENAI API KEY를 입력해주세요

-ChatBot.py에서 GPT의 역할을 당신 마음대로 커스텀할 수 있습니다. 필요한 경우 역할을 바꾸어보세요.
(default role = "You're a very cute lover who reads my feelings and talks to me so kindly")
- 프로그램 실행은 Main.py에서 실행해주세요
  
- 프로그램 실행시 자동으로 당신의 표정을 캠을 통해 확인한 후에 감정을 최초로 1회 분석합니다.
  ![EmoChatExplain](https://github.com/singsingsing3/EmoChat-detecting-your-Emotion-with-cam-and-ChatBot-AI-/assets/120327265/12871d87-9e5a-4db0-affb-72a23bd4f895)

- 'emotion'을 입력하면 추가적으로 다시한번 당신의 표정을 캠을 통해 확인한 후 감정을 다시 분석합니다.
![EmoChatRePredict](https://github.com/singsingsing3/EmoChat-detecting-your-Emotion-with-cam-and-ChatBot-AI-/assets/120327265/c0188390-5311-4b0e-af12-e9a13d9f93c2)

  
- 'exit'를 입력하면 프로그램이 종료됩니다.

- 자유롭게 EmoChat과 채팅하며 위로를 얻어보세요
  ![Emochat_warm](https://github.com/singsingsing3/EmoChat-detecting-your-Emotion-with-cam-and-ChatBot-AI-/assets/120327265/cf7d419c-1157-4314-801f-e58a9d3dc73f)


## ✏️ 자세한 프로그램 설명

### 💬 학습된 모델은 ResNet50을 이용하여 학습했습니다. 자세한 코드는 model_train폴더의 ipynb파일을 확인해주세요
```
# import하여 모델 구현

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import ResNet50

# ResNet50 모델 불러오기 (ImageNet 사전 훈련된 가중치 사용, 최상위 레이어 포함하지 않음)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

# 모델의 출력 레이어 제거 및 새로운 출력 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D(keepdims=True)(x)  # keepdims=True 설정
x = Flatten()(x)  # 1차원으로 평탄화
output = Dense(len(mlb.classes_), activation='softmax')(x)  # 분류할 클래스 수에 맞게 설정

# 새로운 모델 정의
model = Model(inputs=base_model.input, outputs=output)

# 모델 구조 확인
model.summary()

```
### 💬 학습된 클래스는 총 5개로 anger happy normal sad worry를 분류하도록 학습되었습니다.
- 사진은 각 클래스 별로 1200장 씩 학습하였으며 데이터 증강을 이용하였습니다.
- train and test set분리 코드
```
from sklearn.model_selection import train_test_split

seed = 47

(x_train, x_test, y_train, y_test) = train_test_split(
    images, enc_labels, test_size=0.2, random_state=seed
)
print(">> train test shape = {} {}".format(
    x_train.shape, y_train.shape)
)
```
- 데이터 증강 코드
```
# 데이터 증강 설정
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
```

### 💬 모델의 Accuray 결과는 아래 사진과 같습니다.
![ResNet50Acc](https://github.com/singsingsing3/EmoChat-detecting-your-Emotion-with-cam-and-ChatBot-AI-/assets/120327265/82d8d13b-e263-4e68-8c41-629559710f50)


### 💬 개발 환경 설명
- 모델 학습은 구글 Colab을 통하여 학습했습니다.
- 프로그램 실행은 VSCODE를 이용했습니다.
- 환경과 버전은 아래와 같습니다.
```
<코랩 환경>
Python: 3.10.12 
CV: 4.8.0
Numpy: 1.25.2
Tensorflow: 2.15.0
Keras: 2.15.0
```
```
<프로그램 실행환경>
Python: 3.10.14
CV: 4.8.0
Numpy: 1.25.2
Tensorflow: 2.15.0
Keras: 2.15.0
```

### 💬 ChatGPT API 설명

- model="gpt-3.5-turbo" 모델입니다.
- 당신의 API를 이용하여 ChatBot.py에 더 나은 모델로 수정할 수 있습니다.

## ✏️ 참고한 것들
- 서울과학기술대학교 최성록 교수님의 오픈소스 소프트웨어 수업 자료 중 'tkinter' 수업자료를 참고하여 채팅 GUI를 구현했습니다.
  
- 초기 모델학습 스켈레톤 코드는 [참고한 모델학습 코드](https://velog.io/@robert-lee/Tensorflow-Keras-Multi-Class-Classification-%EC%9D%84-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90) 여기서 참고했습니다.
  => 이후 새로운 학습 데이터와 새로운 모델에 맞게 수정하여 사용했습니다.
  
-GPT API 사용은 OPEN AI에서 제공하는 공식 문서를 참고하였습니다.[OPEN AI API 문서](https://platform.openai.com/docs/api-reference/introduction)

-이미지 학습 데이터는 AI HUB에서 제공하는 사진으로 학습했습니다.[AI HUB 표정학습 이미지 데이](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=82)

-추가적으로 GPT 3.5의 도움을 받았습니다.
