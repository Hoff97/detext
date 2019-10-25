import { Injectable } from '@angular/core';
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
    const inputs = [
      new Tensor(new Float32Array([1.0, 2.0, 3.0, 4.0]), 'float32', [2, 2])
    ];
    const outputMap = await this.session.run(inputs);
    const outputTensor = outputMap.values().next().value;
  }

  private async setupModel() {
    const url = 'assets/models/mobile_cnn.onnx';
    await this.session.loadModel(url);
    console.log('yay');
  }
}
