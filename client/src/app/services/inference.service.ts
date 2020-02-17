import { EventEmitter, Injectable } from '@angular/core';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { InferenceSession, Tensor } from 'onnxjs';
import { Model } from '../data/types';
import { base64ToBinary } from '../util/data';
import { ModelService } from './model.service';
import { SettingsService } from './settings.service';
import { SymbolService } from './symbol.service';

@Injectable({
  providedIn: 'root'
})
export class InferenceService {
  private backend: string;

  private decodedModel: Uint8Array;

  private session: InferenceSession;

  public modelAvailable = new EventEmitter<boolean>();
  public model = false;
  public updating = new EventEmitter<boolean>();
  public modelUpdating = true;

  constructor(private modelService: ModelService,
              private symbolService: SymbolService,
              private settingsService: SettingsService) {
    this.setupModelLocal();

    this.setupModel();

    this.backend = this.settingsService.getData().backend;

    this.settingsService.dataChange.subscribe(data => {
      if (this.backend !== data.backend) {
        this.changeBackend(data.backend);
      }
    });
  }

  public async infer(image: ImageData) {
    const input = this.preprocess(image.data, image.width, image.height);

    const outputMap = await this.session.run([input]);
    const output: Tensor = outputMap.values().next().value;

    return this.softMax(output.data as Float32Array);
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

  private async setupModelLocal() {
    const modelPromise = this.modelService.getRecentLocal();
    const model = await modelPromise;
    if (model) {
      await this.setModel(model);
    }
  }

  private async setupModel() {
    const modelPromise = this.modelService.getRecent().toPromise();
    const model = await modelPromise;
    if ((model as any).timestamp) {
      await this.setModel(model as any);
    }
    this.updating.emit(false);
    this.modelUpdating = false;
  }

  private async setModel(model: Model) {
    this.decodedModel = base64ToBinary(model.model);

    this.session = new InferenceSession({ backendHint: this.backend }) ;
    await this.session.loadModel(this.decodedModel);

    this.modelAvailable.emit(true);
    this.model = true;
  }

  private async changeBackend(backend: string) {
    this.backend = backend;
    this.session = new InferenceSession({ backendHint: this.backend }) ;
    await this.session.loadModel(this.decodedModel);
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
