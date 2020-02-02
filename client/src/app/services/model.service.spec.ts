import { TestBed } from '@angular/core/testing';

import { ModelService } from './model.service';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { SettingsService, Settings } from './settings.service';
import { DbService } from './db.service';
import { EventEmitter } from '@angular/core';
import { of } from 'rxjs';

describe('ModelService', () => {
  let spySettings: jasmine.SpyObj<SettingsService>;
  let spyDb: jasmine.SpyObj<DbService>;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    spySettings = jasmine.createSpyObj('SettingsService', ['getData', 'setData']);
    spySettings.dataChange = new EventEmitter<Settings>();
    spyDb = jasmine.createSpyObj('DbService', ['getSymbols', 'saveSymbols', 'saveModel', 'getModel']);

    TestBed.configureTestingModule({
      imports: [
        HttpClientTestingModule,
      ],
      providers: [
        { provide: SettingsService, useValue: spySettings },
        { provide: DbService, useValue: spyDb }
      ]
    });

    httpMock = TestBed.get(HttpTestingController);
  });

  it('should be created', () => {
    const service: ModelService = TestBed.get(ModelService);
    expect(service).toBeTruthy();
  });

  it('should download symbols on change in settings', () => {
    const service: ModelService = TestBed.get(ModelService);

    spyDb.saveModel.and.returnValue(Promise.resolve());

    spySettings.dataChange.emit({
      backend: 'cpu',
      backendAuto: true,
      download: true
    });

    const req = httpMock.expectOne('api/model/latest/?format=json');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should load model locally if network is unavailable', () => {
    const service: ModelService = TestBed.get(ModelService);

    spyDb.getModel.and.returnValue(of({}) as any);

    spySettings.dataChange.emit({
      backend: 'cpu',
      backendAuto: true,
      download: true
    });

    const req = httpMock.expectOne('api/model/latest/?format=json');
    expect(req.request.method).toBe('GET');
    req.flush('', { status: 400, statusText: '' });
  });

  it('should allow retraining', () => {
    let success;

    const service: ModelService = TestBed.get(ModelService);

    service.retrain().subscribe(x => {
      success = true;
    });

    const req = httpMock.expectOne('api/model/train/?format=json');
    expect(req.request.method).toBe('POST');
    req.flush('Ok');

    expect(success).toBeTruthy();
  });
});
