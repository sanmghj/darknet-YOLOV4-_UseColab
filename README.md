# darknet-YOLOV4-_UseColab
Colab을 이용한 darknet(YOLOV4) 훈련 방법 및 실행
# 훈련을 위한 사진 라벨링
- 모델 학습을 위해 라벨링한 객체의 이름을 미리 알려주어야 합니다.
(*주의 폴더 경로에 한글이 존재할 때 오류가 발생합니다!)

-라벨링 도구 다운 및 압축 해제
링크: https://drive.google.com/file/d/1lanO8SyY2QlbVCbOR0LlwQjQYbhoteTd/view

-라벨링 도구 폴더에 data폴더 생성
- 
obj.names의 파일을 만들어줍니다.
<pre>
<code>
예)obj.names 파일의 내용
cat
person
mask
</code>
</person>
위에 처럼 여러개 혹은 한개만 입력할 수 있다.





참고: 
darknet: https://github.com/AlexeyAB/darknet
Yolo_Label: https://github.com/developer0hye/Yolo_Label
코랩이용 참고 사이트: https://ichi.pro/ko/sayongja-jijeong-yolov4-gaeche-tamjigi-gyoyug-google-colab-sayong-6710443722856
