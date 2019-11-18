import { EventEmitter, Injectable } from '@angular/core';
import { wasmSupport } from '../util/util';

export interface Settings {
  backend: 'wasm' | 'cpu';
  backendAuto: boolean;
  download: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class SettingsService {
  private static localStorageKey = 'settings';

  private data: Settings;

  public dataChange = new EventEmitter<Settings>();

  constructor() {
    let d = JSON.parse(localStorage.getItem(SettingsService.localStorageKey));
    if (d === null) {
      d = {};
    }
    if (d.backendAuto === undefined) {
      d.backendAuto = true;
    }
    if (d.backend === undefined && d.backendAuto) {
      d.backend = wasmSupport() ? 'wasm' : 'cpu';
    }
    if (d.download === undefined) {
      d.download = true;
    }

    this.setData(d);
  }

  public setData(data: Settings) {
    this.data = data;

    if (this.data.backendAuto) {
      this.data.backend = wasmSupport() ? 'wasm' : 'cpu';
    }

    this.dataChange.emit(this.data);

    localStorage.setItem(SettingsService.localStorageKey, JSON.stringify(this.data));
  }

  public getData() {
    return this.data;
  }
}
