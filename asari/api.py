import pathlib

import onnxruntime as rt

from asari.preprocess import tokenize


class Sonar:
    def __init__(self):
        pipeline_file = pathlib.Path(
            __file__).parent / "data" / "pipeline.onnx"
        self.sess = rt.InferenceSession(str(pipeline_file))
        self.input_name = self.sess.get_inputs()[0].name
        self.prob_name = self.sess.get_outputs()[1].name

        input_name = self.sess.get_inputs()[0].name
        print("Input name  :", input_name)
        input_shape = self.sess.get_inputs()[0].shape
        print("Input shape :", input_shape)
        input_type = self.sess.get_inputs()[0].type
        print("Input type  :", input_type)

        output_name = self.sess.get_outputs()[0].name
        print("Output name  :", output_name)
        output_shape = self.sess.get_outputs()[0].shape
        print("Output shape :", output_shape)
        output_type = self.sess.get_outputs()[0].type
        print("Output type  :", output_type)

    def ping(self, text: str):
        tokenized = tokenize(text)
        proba = self.sess.run(
            [self.prob_name], {self.input_name: [tokenized]})[0][0]
        print(tokenized, proba)
        res = {
            "text": text,
            "top_class": max(proba, key=lambda k: proba[k]),
            "classes": [
                {"class_name": class_name, "confidence": confidence} for class_name, confidence in proba.items()
            ],
        }
        return res


if __name__ == "__main__":
    sonar = Sonar()
    # print(sonar.input_name, sonar.prob_name)
    print(sonar.ping("今日はいい天気ですね"))
