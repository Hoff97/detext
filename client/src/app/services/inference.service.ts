import { Injectable } from '@angular/core';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { InferenceSession, Tensor } from 'onnxjs';
import { ModelService } from './model.service';
import { SymbolService } from './symbol.service';

@Injectable({
  providedIn: 'root'
})
export class InferenceService {

  // GPU Backend seems to be broken right now, so CPU will be used
  private session = new InferenceSession({ backendHint: 'cpu' });

  private classes: string[];

  constructor(private modelService: ModelService, private symbolService: SymbolService) {
    this.setupModel();

    this.symbolService.getSymbols().subscribe((symbols) => {
      this.classes = symbols.map(symbol => symbol.name);
    });
  }

  public async infer(image: ImageData) {
    const inputs = this.preprocess(image.data, image.width, image.height);
    const outputMap = await this.session.run([inputs]);
    const outputTensor: Tensor = outputMap.values().next().value;
    return this.softMax(outputTensor.data as Float32Array);
  }

  public getClasses() {
    return this.classes;
  }

  private softMax(array: Float32Array) {
    let sum = 0;
    for (let i = 0; i < array.length; i++) {
      array[i] = Math.exp(array[i]);
      sum += array[i];
    }
    for (let i = 0; i < array.length; i++) {
      array[i] = array[i] / sum;
    }
    return array;
  }

  private async setupModel() {
    const model = await this.modelService.getRecent().toPromise();
    const decoded = this.convertDataURIToBinary(model.model);
    await this.session.loadModel(decoded);
  }

  private convertDataURIToBinary(data: string) {
    const raw = atob(data);
    const rawLength = raw.length;
    const array = new Uint8Array(new ArrayBuffer(rawLength));
    for (let i = 0; i < rawLength; i++) {
      array[i] = raw.charCodeAt(i);
    }
    return array;
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
}
