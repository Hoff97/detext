import { TestBed } from '@angular/core/testing';

import { DbService } from './db.service';
import { SettingsService, Settings, defaultSettings } from './settings.service';
import { EventEmitter } from '@angular/core';

describe('DbService', () => {
  let settingsServiceSpy: jasmine.SpyObj<SettingsService>;

  beforeEach(() => {
    settingsServiceSpy = jasmine.createSpyObj('SettingsService', ['getData', 'setData']);

    settingsServiceSpy.dataChange = new EventEmitter<Settings>();

    TestBed.configureTestingModule({
      providers: [
        { provide: SettingsService, useValue: settingsServiceSpy }
      ]
    });
  });

  it('should be created', () => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    const service: DbService = TestBed.get(DbService);
    expect(service).toBeTruthy();
  });

  it('should clear data if downloads are disabled', () => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    const service: DbService = TestBed.get(DbService);

    settingsServiceSpy.dataChange.emit({
      ...defaultSettings,
      download: false
    });
  });

  it('should allow saving a model', () => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    const service: DbService = TestBed.get(DbService);

    service.saveModel({} as any);
  });

  it('should allow saving symbols', () => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    const service: DbService = TestBed.get(DbService);

    service.saveSymbols([{} as any]);
  });

  it('should allow getting symbols', () => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    const service: DbService = TestBed.get(DbService);

    service.getSymbols();
  });

  afterAll(() => {
    settingsServiceSpy.getData.and.returnValue(defaultSettings);

    settingsServiceSpy.dataChange.emit({
      ...defaultSettings,
      download: false
    });
  });
});
