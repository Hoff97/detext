import { Injectable } from '@angular/core';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { Tensor } from 'onnxjs';
import { ClassSymbol } from '../data/types';
import { base64ToBinary } from '../util/data';
import { Inference, MainThreadInference } from './inference/inference';
import { ModelService } from './model.service';
import { SettingsService } from './settings.service';
import { SymbolService } from './symbol.service';

@Injectable({
  providedIn: 'root'
})
export class InferenceService {
  private inference: Inference;

  private classes: ClassSymbol[];

  private backend: string;

  private decodedModel: Uint8Array;

  constructor(private modelService: ModelService,
              private symbolService: SymbolService,
              private settingsService: SettingsService) {
    this.setupModel();

    this.symbolService.getSymbols().subscribe((symbols) => {
      this.classes = symbols.map(symbol => symbol);
    });

    this.backend = this.settingsService.getData().backend;

    this.settingsService.dataChange.subscribe(data => {
      if (this.backend !== data.backend) {
        this.backend = data.backend;
        this.inference = new MainThreadInference(this.decodedModel, this.backend);
      }
    });
  }

  public async infer(image: ImageData) {
    const inputs = this.preprocess(image.data, image.width, image.height);
    const output = await this.inference.infer(inputs);
    return this.softMax(output.data as Float32Array);
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
    this.decodedModel = base64ToBinary(model.model);

    this.inference = new MainThreadInference(this.decodedModel, this.backend);
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
