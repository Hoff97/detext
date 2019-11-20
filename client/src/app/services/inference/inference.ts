import { InferenceSession, Tensor } from 'onnxjs';

export interface Inference {
  infer(input: Tensor): Promise<Tensor>;

  setModel(model: Uint8Array);
}

export class MainThreadInference implements Inference {
  private session: InferenceSession;

  constructor(model: Uint8Array, backend: string) {
    this.session = new InferenceSession({ backendHint: backend }) ;
    this.session.loadModel(model);
  }

  async infer(input: Tensor): Promise<Tensor> {
    const outputMap = await this.session.run([input]);
    const outputTensor: Tensor = outputMap.values().next().value;
    return outputTensor;
  }

  async setModel(model: Uint8Array) {
    await this.session.loadModel(model);
  }
}
