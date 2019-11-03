import { Injectable } from '@angular/core';
import Dexie from 'dexie';
import { Model } from './model.service';


@Injectable({
  providedIn: 'root'
})
export class DbService extends Dexie {

  models: Dexie.Table<Model, number>;

  constructor() {
    super('detext');

    this.version(1).stores({
      model: '++id, timestamp, model'
    });

    this.models = this.table('model');
  }

  async getModel(): Promise<Model> {
    return await this.models.get(1);
  }

  async saveModel(model: Model) {
    await this.models.delete(1);
    model.id = 1;
    await this.models.add(model);
  }
}
