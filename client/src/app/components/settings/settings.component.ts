import { Location } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Settings, SettingsService } from 'src/app/services/settings.service';
import { LoginService } from 'src/app/services/login.service';
import { ModelService } from 'src/app/services/model.service';

@Component({
  selector: 'app-settings',
  templateUrl: './settings.component.html',
  styleUrls: ['./settings.component.css']
})
export class SettingsComponent implements OnInit {

  public data: Settings;

  public loggedIn: boolean;
  public retraining = false;

  public backends = [
    { value: 'wasm', name: 'Web Assembly' },
    { value: 'cpu', name: 'Javascript'}
  ];

  constructor(private settingsService: SettingsService,
              private location: Location,
              private loginService: LoginService,
              private modelService: ModelService) {
    this.data = settingsService.getData();
    this.settingsService.dataChange.subscribe(data => {
      this.data = data;
    });
    this.loggedIn = this.loginService.isLoggedIn();

    this.loginService.loginSucceeded.subscribe(ev => {
      this.loggedIn = true;
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

  retrain() {
    this.retraining = true;
    this.modelService.retrain().subscribe(ev => {
      this.retraining = false;
      console.log('Retrained');
    });
  }
}
