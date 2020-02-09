import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SettingsComponent } from './settings.component';
import { FormsModule } from '@angular/forms';
import { LoginComponent } from '../login/login.component';
import { RouterTestingModule } from '@angular/router/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { LoginService } from 'src/app/services/login.service';
import { ModelService } from 'src/app/services/model.service';
import { SettingsService, Settings, defaultSettings } from 'src/app/services/settings.service';
import { EventEmitter } from '@angular/core';
import { NEVER, of } from 'rxjs';

describe('SettingsComponent', () => {
  let component: SettingsComponent;
  let fixture: ComponentFixture<SettingsComponent>;
  let loginServiceSpy: jasmine.SpyObj<LoginService>;
  let modelServiceSpy: jasmine.SpyObj<ModelService>;
  let settingsServiceSpy: jasmine.SpyObj<SettingsService>;

  beforeEach(async(() => {
    loginServiceSpy = jasmine.createSpyObj('LoginService', ['login', 'getToken', 'isLoggedIn']);
    settingsServiceSpy = jasmine.createSpyObj('SettingsService', ['getData', 'setData']);
    modelServiceSpy = jasmine.createSpyObj('ModelService', ['retrain']);

    settingsServiceSpy.dataChange = new EventEmitter<Settings>();
    loginServiceSpy.loginSucceeded = new EventEmitter<void>();

    TestBed.configureTestingModule({
      declarations: [ SettingsComponent, LoginComponent ],
      imports: [ FormsModule, RouterTestingModule, HttpClientTestingModule ],
      providers: [
        { provide: LoginService, useValue: loginServiceSpy },
        { provide: ModelService, useValue: modelServiceSpy },
        { provide: SettingsService, useValue: settingsServiceSpy }
      ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    fixture = TestBed.createComponent(SettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should handle settings changes', () => {
    settingsServiceSpy.dataChange.emit({
      ...defaultSettings,
      backend: 'wasm'
    });

    expect(component.data.backend).toEqual('wasm');
  });

  it('should handle logins', () => {
    loginServiceSpy.loginSucceeded.emit();

    expect(component.loggedIn).toBeTruthy();
  });

  it('should handle backend changes', () => {
    component.changeBackendAuto(false);
  });

  it('should handle backend changes 2', () => {
    component.changeBackend('cpu');
  });

  it('should handle download changes', () => {
    component.changeDownload(false);
  });

  it('should handle going back', () => {
    component.back();
  });

  it('should handle retraining', () => {
    modelServiceSpy.retrain.and.returnValue(NEVER);

    component.retrain();

    expect(component.retraining).toBeTruthy();
  });

  it('should handle retraining success', () => {
    modelServiceSpy.retrain.and.returnValue(of({}) as any);

    component.retrain();

    expect(component.retraining).toBeFalsy();
  });
});
