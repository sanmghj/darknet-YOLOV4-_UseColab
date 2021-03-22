# darknet-YOLOV4-_UseColab
Colab을 이용한 darknet(YOLOV4) 훈련 방법 및 실행

----------

##Colab 설정   
 1. Colab사이트 접속
 링크: https://colab.research.google.com/notebooks/welcome.ipynb?hl=ko
 2. 내 노트 생성 
 3. 수정 -> 노트설정 ->  하드웨어 가속기를 GPU로 변경
 <img heights="150" width="200" alt="label1" src="https://user-images.githubusercontent.com/38642688/111928915-59ea9380-8af8-11eb-8a86-a0f5087c8cb9.PNG">

##Google drive 설정

 - Google drive에 yolov4 폴더를 생성한다.
 - yolov4폴더로 이동 training 폴더를 생성한다.   
 <img heights="150" width="200" alt="label1" src="https://user-images.githubusercontent.com/38642688/111929479-c1eda980-8af9-11eb-9c81-dcd35ab5a8b1.PNG">

----------

##Darknet 설치
 1. Colab으로 이동 Google drive를 마운트 한다.
 2. 코드
>` %cd ..
from google.colab import drive
drive.mount('/content/gdrive')`

 3. 심볼릭 링크를 만들어 경로를 설정한다.
>  `!ln -s /content/gdrive/My\ Drive/ /mydrive
%cd /mydrive/yolov4`

 4.  git에서 darknet폴더를 복제한다.
> `!git clone https://github.com/AlexeyAB/darknet`

 5. darknet폴더 이동
> `%cd darknet  ` 

 6. 현재까지 설정확인
 <img heights="150" width="200" alt="label1" src="https://user-images.githubusercontent.com/38642688/111929658-59eb9300-8afa-11eb-9457-5b88d80c8b87.PNG">

---------- 

## 훈련을 위한 파일 생성  
- 라벨링 도구 다운 및 압축 해제   
링크: https://drive.google.com/file/d/1lanO8SyY2QlbVCbOR0LlwQjQYbhoteTd/view   
(*주의 폴더 경로에 한글이 존재할 때 오류가 발생합니다!)   
- 라벨링 도구 폴더에 data 폴더 생성   
#####1. obj.names 파일 준비
- data 폴더에 obj.names 파일 생성

> 예)obj.names 파일의 내용
> `cat
person
mask `


###### 위에 처럼 여러개 혹은 한개만 입력할 수 있다. 
#####2. obj.data 파일 준비
 - data폴더에 obj.data 파일 생성

> `classes = (훈련시킬 대상의 갯수)
train = data/train.txt
valid = data/test.txt
names = data/obj.names
backup = /mydrive/yolov4/training`
  
##### 3. 훈련 이미지 준비
- data 폴더에 image 폴더 생성   
- image 폴더에 학습할 이미지 저장   
- YoloLabel.exe 실행
- Open Files -> 이미지가 저장되있는 폴더 지정(data/image)
<img heights="300" width="400" alt="label1" src="https://user-images.githubusercontent.com/38642688/111928270-75ed3580-8af6-11eb-8447-0c74219bc271.png">
- 메세지 내용에 따라 obj.names지정
<img heights="300" width="400" alt="label2" src="https://user-images.githubusercontent.com/38642688/111928274-7685cc00-8af6-11eb-9052-7ad8d71c681c.png">   
- Name별 이미지내 객체 위치를 지정하고 Save를 눌러 저장
<img heights="300" width="400" alt="label3" src="https://user-images.githubusercontent.com/38642688/111928275-771e6280-8af6-11eb-8dd6-9e76e08704c5.png">
- 라벨링을 마친 image폴더를 압축한다.

#####4. 훈련을 위한 yolov4-custom.cfg파일 수정

 - Google drive yolov4\darknet\cfg폴더내의 yolov4-custom.cfg파일 다운로드
 <img heights="300" width="400" alt="label1" src="https://user-images.githubusercontent.com/38642688/111929803-c23a7480-8afa-11eb-8b4c-1003e83fa246.PNG">
 - 다운 받은 cfg파일을 열고 Colab사양에 맞게 값을 변경한다.
 - [yolo]부분과 [convolutional]부분 값 변경
총 3곳 변경이므로 유의한다!
classes는 훈련시킬 대상의 갯수이다.

>`[net]
>batch = 64 
> subdivision = 16
> width = 416
> height = 416
> max_batches = (classes * 2000)
> steps = (max_batches의 80%값), (max_batches의 90%값)` 
> 
> `[yolo] 
> filter = ( classes + 5 ) * 3`
> 해당 라인은 961, 1049, 1137라인이다.
>
>`[convolutional]
classes = 훈련시킬 대상의 갯수`
해당라인은 968, 1049, 1144라인이다.

#####5. 훈련이미지 경로 생성을 위한 process.py 파이썬 파일 다운 - 
 - 링크: https://drive.google.com/file/d/1v_nqcIERTTB7ke9oY2sG8TwJN5S8FX5W/view

##### 6. 준비된 5개의 파일들을 yolov4 폴더에 넣어준다.
<img heights="300" width="600" alt="label1" src="https://user-images.githubusercontent.com/38642688/111932058-f9f7eb00-8aff-11eb-98e9-153e923491af.PNG">

 - 위 사진처럼 파일이 올라가 있어야한다.
 
----------

## 훈련을 위한 설정
 1. Colab으로 이동
 2. 메이크 파일 변경하여 opencv 및 GPU 활성화
> ` !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    > !sed -i 's/GPU=0/GPU=1/' Makefile
    > !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    > !sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    > !sed -i 's/LIBSO=0/LIBSO=1/' Makefile`

 3. 메이크 실행
> `!make`

 4. Google drive에 올린 파일 압축 해제 및 설정
>- 이미지 파일 압축 해제
> `!unzip /mydrive/yolov4/image.zip -d data/image`
>- yolov4-custom.cfg, obj.names, obj.data 파일 복사
> `!cp /mydrive/yolov4/yolov4-custom.cfg cfg
>!cp /mydrive/yolov4/obj.names data 
>!cp /mydrive/yolov4/obj.data  data`
>- process.py파일 실행
>`!cp /mydrive/yolov4/process.py .
!python process.py`

 5. 현재까지 파일 데이터 확인
> `!ls data/`

 6. 사전 정의 된 yolov4 가중치 다운로드
> `!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137`


----------

## 훈련하기
 1.  darknet 훈련 코드 실행
> `!./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map`

 - 접속 유지를 위한 세션 방지 코드입력 

 2. Ctrl + shift + i를 누르고 Console에 다음 코드 입력후 Enter
> `	function ClickConnect(){
	console.log("Working"); 
	document
	  .querySelector('#top-toolbar > colab-connect-button')
	  .shadowRoot.querySelector('#connect')
	  .click() 
	}
setInterval(ClickConnect,60000)`

##결과확인

 1. 결과 확인을 위한 imShow 함수 추가
	 > ``` python
> def imShow(path):
>    import cv2
>    import matplotlib.pyplot as plt
>    %matplotlib inline
>    
>    image = cv2.imread(path)
>    height, width = image.shape[:2]
>    resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
>    
>    fig = plt.gcf()
>    fig.set_size_inches(18, 10)
>    plt.axis("off")
>    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
>    plt.show()
```

 2. 결과 차트 확인
 3. mAP(평균 정밀도) 확인
 4. 학습 사진을 제외한 다른 사진을 이용해 결과 확인


 

#####참고:    
darknet: https://github.com/AlexeyAB/darknet   
Yolo_Label: https://github.com/developer0hye/Yolo_Label   
코랩이용 참고 사이트: https://ichi.pro/ko/sayongja-jijeong-yolov4-gaeche-tamjigi-gyoyug-google-colab-sayong-6710443722856   


