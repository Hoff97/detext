import { InferenceSession, Tensor } from 'onnxjs';

export interface Inference {
  infer(input: Tensor): Promise<Tensor>;
}

export class MainThreadInference implements Inference {
  private session = new InferenceSession({ backendHint: 'wasm' });

  constructor(model: Uint8Array) {
    this.session.loadModel(model);
  }

  async infer(input: Tensor): Promise<Tensor> {
    const outputMap = await this.session.run([input]);
    const outputTensor: Tensor = outputMap.values().next().value;
    return outputTensor;
  }
}
