import { Injectable } from '@angular/core';
import Dexie from 'dexie';
import { ClassSymbol, Model } from '../data/types';
import { Settings, SettingsService } from './settings.service';


@Injectable({
  providedIn: 'root'
})
export class DbService extends Dexie {

  models: Dexie.Table<Model, number>;
  symbols: Dexie.Table<ClassSymbol, number>;

  private store: boolean;

  constructor(private settingsService: SettingsService) {
    super('detext');

    this.version(1).stores({
      model: '++id, timestamp, model',
      symbol: '++id, name, timestamp'
    });

    this.models = this.table('model');
    this.symbols = this.table('symbol');

    this.store = settingsService.getData().download;
    this.settingsService.dataChange.subscribe((data: Settings) => {
      this.store = data.download;
      if (!this.store) {
        this.clearAll();
      }
    });
  }

  async getModel(): Promise<Model> {
    return (await this.models.toArray())[0];
  }

  async saveModel(model: Model) {
    await this.models.clear();
    await this.models.add(model);
  }

  async saveSymbols(symbols: ClassSymbol[]) {
    await this.symbols.clear();
    await this.symbols.bulkAdd(symbols);
  }

  async getSymbols(): Promise<ClassSymbol[]> {
    return this.symbols.orderBy('id').toArray();
  }

  async clearAll() {
    await this.models.clear();
    await this.symbols.clear();
  }
}
