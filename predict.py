"""Sample module for predicting face marks with HRNetV2."""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf
import math
from postprocessing import parse_heatmaps, draw_marks
from preprocessing import normalize
from face_detector.detector import Detector
from quantization import TFLiteModelPredictor
import wave
import pygame
import os

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--write_video", type=bool, default=True,
                    help="Write output video.")
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Allow GPU memory growth.
#devices = tf.config.list_physical_devices('GPU')
#for device in devices:
#    tf.config.experimental.set_memory_growth(device, True)


def alarm_On():



    # 알람음을 재생합니다.

    pygame.mixer.music.play()




def alarm_Off():

    # 알람음을 중지합니다.

    pygame.mixer.music.stop()

file_path = 'nomal_alarm.wav'

file_wav = wave.open(file_path)

frequency = file_wav.getframerate()

pygame.mixer.init(frequency=frequency)

pygame.mixer.music.load(file_path)

if __name__ == "__main__":

    count = 0
    alarm_count = 0

    threshold = 0.7

    # 얼굴 감지기 로드
    detector_face = Detector('assets/face_model')

    # 모델 로드
    model = tf.keras.models.load_model("./exported/hrnetv2")
    #model = tf.keras.models.load_model("test_model_batchsize24")
    #model = TFLiteModelPredictor(
    #     "optimized/hrnet_quant_int_only.tflite")


    #video_src = args.cam if args.cam is not None else args.video
    video_src = "졸음테스트.mp4"
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # 소스가 없으면 웹캠이용
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # 실제 프레임 크기 저장
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # 비디오 저장시 사용
    if args.write_video:
        video_writer = cv2.VideoWriter(
            'output.mp4', cv2.VideoWriter_fourcc(*'X264'), frame_rate//4, (frame_width, frame_height))

    # Introduce a metter to measure the FPS.
    tm = cv2.TickMeter()


    while True:
        tm.start()


        frame_got, frame = cap.read()
        if frame_got is False:
            break



        # 웹캠 이용시 거울효과, 이미지를 반전시킴
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # 이미지 전처리, 얼굴감지기로 감지한 얼굴을 자름
        _image = detector_face.preprocess(frame)

        #
        boxes, scores, _ = detector_face.predict(_image, threshold)


        boxes = detector_face.transform_to_square(
            boxes, scale=1.22, offset=(0, 0.13))


        boxes, _ = detector_face.clip_boxes(
            boxes, (0, 0, frame_height, frame_width))
        boxes = boxes.astype(np.int32)

        if boxes.size > 0:
            faces = []
            for facebox in boxes:

                top, left, bottom, right = facebox
                face_image = frame[top:bottom, left:right]


                face_image = cv2.resize(face_image, (256, 256))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_image = normalize(np.array(face_image, dtype=np.float32))
                faces.append(face_image)

            faces = np.array(faces, dtype=np.float32)


            heatmap_group = model.predict(faces)


            mark_group = []
            heatmap_grids = []
            for facebox, heatmaps in zip(boxes, heatmap_group):
                top, left, bottom, right = facebox
                width = height = (bottom - top)

                marks, heatmap_grid = parse_heatmaps(heatmaps, (width, height))


                marks[:, 0] += left
                marks[:, 1] += top

                mark_group.append(marks)
                heatmap_grids.append(heatmap_grid)


            # EAR
            # mark_group[0] = mark_group[0][60:76] 가로 60, 64 ,,,, 62, 66
            p62X = mark_group[0][62][0]
            p62Y = mark_group[0][62][1]
            p66X = mark_group[0][66][0]
            p66Y = mark_group[0][66][1]
            p60X = mark_group[0][60][0]
            p60Y = mark_group[0][60][1]
            p64X = mark_group[0][64][0]
            p64Y = mark_group[0][64][1]

            p60p64 = math.sqrt(((p64X - p60X)**2 + (p64Y - p60Y)**2))
            #print("가로: ", p60p64)
            p62p66 = math.sqrt(((p62X - p66X)**2 + (p62Y - p66Y)**2))
            #print("세로: ", p62p66)
            EAR_THRESH_HOLD = p62p66/p60p64
            #print("EAR: ", EAR_THRESH_HOLD)
            cv2.putText(frame, 'EAR: {:.2f}'.format(EAR_THRESH_HOLD), (0,frame_height), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))
            #cv2.putText(frame, 'EAR: {:.2f}'.format(EAR_THRESH_HOLD), (frame_width//2,frame_height//2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0))



            if EAR_THRESH_HOLD < 0.3:
                count += 1

            else:
                count = count -1
            if count < 0:
                count = 0

            #print(count)
            if count > 30:

                cv2.putText(frame, 'Alarm', (frame_height//2, frame_width//2), cv2.FONT_HERSHEY_DUPLEX, 10.0, (255,255,0))
                alarm_On()
                alarm_count += 1
                if alarm_count > 30:
                    alarm_Off()
                    alarm_count = 0

            #print(frame_rate)






            # 바운딩박스 및 특징점 출력
            draw_marks(frame, mark_group)

            detector_face.draw_boxes(frame, boxes, scores)

            # 첫번쨰 히트맵을 출력하고싶으면
            #cv2.imshow("heatmap_grid", heatmap_grid[0])

        # 결과 출력
        cv2.imshow('image', frame)





        if args.write_video:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:
            break
