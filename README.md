# Drowsiness-driving-detection

https://wywu.github.io/projects/LAB/WFLW.html

1. 위의 링크에서 WFLW 데이터셋을 받고 preprocessing폴더에 wflw폴더를 만든 후 WFLW_annotations, WFLW_annotations를 옮겨줌이후     generate_mesh_dataset.py 123줄 경로설정
2. generate_mesh_dataset.py 의 266줄 TRUE -> train, False -> test
3. wflw_cropped 을 train.py와 같은경로로 옮겨줌
4. wflw_cropped에 전처리된 데이터셋을 넣어줌
5. train.py 실행
6. 저장된 모델을 exported/hrnetv2에 넣어줌
7. predict.py 실행
