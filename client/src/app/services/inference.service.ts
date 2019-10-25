import { Injectable } from '@angular/core';
import ndarray from 'ndarray';
import { InferenceSession, Tensor } from 'onnxjs';

@Injectable({
  providedIn: 'root'
})
export class InferenceService {

  private session = new InferenceSession();

  constructor() {
    this.setupModel();
  }

  public async infer(image: ImageData) {
    const inputs = this.preprocess(image.data, image.width, image.height);
    const outputMap = await this.session.run([inputs]);
    const outputTensor = outputMap.values().next().value;
    console.log(outputTensor);
  }

  private async setupModel() {
    const url = 'assets/models/mobile_cnn.onnx';
    await this.session.loadModel(url);
  }

  private preprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

    dataProcessed.pick(0, 0, null, null).set(dataFromImage.pick(null, null, 0));
    dataProcessed.pick(0, 1, null, null).set(dataFromImage.pick(null, null, 1));
    dataProcessed.pick(0, 2, null, null).set(dataFromImage.pick(null, null, 2));

    const tensor = new Tensor(new Float32Array(3 * width * height), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessed.data);
    return tensor;
  }
}
