import { Injectable } from '@angular/core';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
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

    ops.assigns(dataProcessed, 1.0);

    const op = ndarray(new Float32Array(width * height * 4), [width, height, 4]);
    ops.assigns(op, 255);
    ops.div(dataFromImage, dataFromImage, op);

    ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
    ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

    const tensor = new Tensor(new Float32Array(3 * width * height), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessed.data);
    return tensor;
  }

  private test(data) {
    let c = 0;
    for (let i = 0; i < data.data.length; i++) {
      if (data.data[i] === 1.0) {
        console.log(i, data.data[i]);
        c++;
      }
      if (c > 10) {
        break;
      }
    }
  }
}
