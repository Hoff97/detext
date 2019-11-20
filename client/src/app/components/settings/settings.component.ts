import { Location } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Settings, SettingsService } from 'src/app/services/settings.service';

@Component({
  selector: 'app-settings',
  templateUrl: './settings.component.html',
  styleUrls: ['./settings.component.css']
})
export class SettingsComponent implements OnInit {

  public data: Settings;

  public backends = [
    { value: 'wasm', name: 'Web Assembly' },
    { value: 'cpu', name: 'Javascript'}
  ];

  constructor(private settingsService: SettingsService,
              private location: Location) {
    this.data = settingsService.getData();
    this.settingsService.dataChange.subscribe(data => {
      this.data = data;
    });
  }

  ngOnInit() {
  }

  changeBackendAuto(value: boolean) {
    this.data.backendAuto = value;
    this.settingsService.setData(this.data);
  }

  changeBackend(value: 'wasm' | 'cpu') {
    this.data.backend = value;
    this.settingsService.setData(this.data);
  }

  changeDownload(value: boolean) {
    this.data.download = value;
    this.settingsService.setData(this.data);
  }

  back() {
    this.location.back();
  }
}
