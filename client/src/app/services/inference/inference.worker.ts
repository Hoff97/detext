/// <reference lib="webworker" />
import * as Comlink from 'comlink';
import { InferenceSession, Tensor } from 'onnxjs';
import { Inference } from './inference';

class WorkerInference implements Inference {
  private session = new InferenceSession({ backendHint: 'cpu' });

  constructor(model: Uint8Array) {
    this.session.loadModel(model);
    console.log('Worker constructed');
  }

  async infer(input: Tensor): Promise<Tensor> {
    const outputMap = await this.session.run([input]);
    const outputTensor: Tensor = outputMap.values().next().value;
    return outputTensor;
  }

}
Comlink.expose(WorkerInference);
