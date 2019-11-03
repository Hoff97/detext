import { Injectable } from '@angular/core';
import Dexie from 'dexie';
import { ClassSymbol, Model } from '../data/types';


@Injectable({
  providedIn: 'root'
})
export class DbService extends Dexie {

  models: Dexie.Table<Model, number>;
  symbols: Dexie.Table<ClassSymbol, number>;

  constructor() {
    super('detext');

    this.version(1).stores({
      model: '++id, timestamp, model',
      symbol: '++id, name, timestamp'
    });

    this.models = this.table('model');
    this.symbols = this.table('symbol');
  }

  async getModel(): Promise<Model> {
    return await this.models.get(1);
  }

  async saveModel(model: Model) {
    await this.models.delete(1);
    model.id = 1;
    await this.models.add(model);
  }

  async saveSymbols(symbols: ClassSymbol[]) {
    await this.symbols.clear();
    for (let i = 0; i < symbols.length; i++) {
      symbols[i].id = i;
    }
    await this.symbols.bulkAdd(symbols);
  }

  async getSymbols(): Promise<ClassSymbol[]> {
    return this.symbols.orderBy('id').toArray();
  }
}
