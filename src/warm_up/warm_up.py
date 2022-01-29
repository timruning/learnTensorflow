import tensorflow as tf
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2


def main():
    with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:
        # replace <request> with one of:
        #
        # classification_pb2.ClassificationRequest(..)
        # regression_pb2.RegressionRequest(..)
        # inference_pb2.MultiInferenceRequest(..)

        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name="youtube_dnn", signature_name="serving_default"),
            inputs={
                "sample": tf.compat.v1.make_tensor_proto([[1, 2, 3, 4, 5]], dtype=tf.int64),
                "label": tf.compat.v1.make_tensor_proto([[0]], dtype=tf.int64)
            }
        )
        print(request)

        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        for r in range(100):
            writer.write(log.SerializeToString())
        writer.write(log.SerializeToString())


if __name__ == "__main__":
    main()
