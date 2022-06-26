"""The training script for HRNet facial landmark detection.
"""
import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from callbacks import EpochBasedLearningRateSchedule, LogImages
from dataset import build_dataset
from network import hrnet_v2
#import tensorflow.compat.v1 as tf2

parser = ArgumentParser()
parser.add_argument("--epochs", default=80, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=16, type=int,
                    help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
parser.add_argument("--eval_only", default=False, type=bool,
                    help="Evaluate the model without training.")
args = parser.parse_args()



#tf2.disable_v2_behavior()  #disable for tensorFlow V2
#physical_devices = tf2.config.experimental.list_physical_devices('GPU')
#tf2.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":



    name = "hrnetv2"

    # 얼굴 랜드마크 특징점 갯수
    number_marks = 98

    # train 데이터 경로
    train_files_dir = "wflw_cropped/train"

    # test 데이터 경로
    test_files_dir = "wflw_cropped/test"


    val_files_dir = None


    sample_image = "docs/face.jpg"



    # 체크포인트 파일 경로
    checkpoint_dir = os.path.join("checkpoints", name)

    # 모델 저장경로
    export_dir = os.path.join("exported", name)

    # log파일 저장경로
    log_dir = os.path.join("logs", name)

    # 인풋이미지 사이즈, hrnet 세팅
    input_shape = (256, 256, 3)
    model = hrnet_v2(input_shape=input_shape, output_channels=number_marks,
                     width=18, name=name)

    # Model built. Restore the latest model if checkpoints are available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")

    # If the restored model is ready for inference, save it and quit training.
    if args.export_only:
        if latest_checkpoint is None:
            print("Warning: Model not restored from any checkpoint.")
        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir)
        print("Model saved at: {}".format(export_dir))
        quit()

    # 테스트셋 빌드
    dataset_test = build_dataset(test_files_dir, "test",
                                 number_marks=number_marks,
                                 image_shape=input_shape,
                                 training=False,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 prefetch=tf.data.experimental.AUTOTUNE)


    if args.eval_only:
        model.evaluate(dataset_test)
        quit()



    #모델 컴파일
    model.compile(optimizer=keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])


    #학습률 세팅
    schedule = [(1, 0.001),
                (30, 0.0001),
                (50, 0.00001)]

    #체크포인트 파일 생성
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, name),
        save_weights_only=True,
        verbose=1,
        save_best_only=True)

    #텐서보드
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1024,
                                                       write_graph=True,
                                                       update_freq='epoch')

    callback_lr = EpochBasedLearningRateSchedule(schedule)


    callback_image = LogImages(log_dir, sample_image)


    callbacks = [callback_checkpoint, callback_tensorboard, callback_lr,
                 callback_image]

    #train 데이터 빌드
    dataset_train = build_dataset(train_files_dir, "train",
                                  number_marks=number_marks,
                                  image_shape=input_shape,
                                  training=True,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  prefetch=tf.data.experimental.AUTOTUNE)

    #val데이터 -> train에서 512개마다 추출
    if val_files_dir:
        dataset_val = build_dataset(val_files_dir, "validation",
                                    number_marks=number_marks,
                                    image_shape=input_shape,
                                    training=False,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    prefetch=tf.data.experimental.AUTOTUNE)
    else:
        dataset_val = dataset_train.take(int(512/args.batch_size))
        dataset_train = dataset_train.skip(int(512/args.batch_size))

    #학습시작
    model.fit(dataset_train,
              validation_data=dataset_val,
              epochs=args.epochs,
              callbacks=callbacks,
              initial_epoch=args.initial_epoch)

    # 모델평가
    model.evaluate(dataset_test)
    #저장
    model.save('test_model_wflw_batchsize32_epochs80')
