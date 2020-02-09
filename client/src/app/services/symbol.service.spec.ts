import { TestBed } from '@angular/core/testing';

import { SymbolService } from './symbol.service';

import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { SettingsService, Settings } from './settings.service';
import { DbService } from './db.service';
import { EventEmitter } from '@angular/core';
import { of } from 'rxjs';

describe('SymbolService', () => {
  let spySettings: jasmine.SpyObj<SettingsService>;
  let spyDb: jasmine.SpyObj<DbService>;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    spySettings = jasmine.createSpyObj('SettingsService', ['getData', 'setData']);
    spySettings.dataChange = new EventEmitter<Settings>();
    spyDb = jasmine.createSpyObj('DbService', ['getSymbols', 'saveSymbols']);

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
    const service: SymbolService = TestBed.get(SymbolService);
    expect(service).toBeTruthy();
  });

  it('should download symbols on change in settings', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    spySettings.dataChange.emit({
      backend: 'cpu',
      backendAuto: true,
      download: true
    });

    const req = httpMock.expectOne('api/symbol/?format=json');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should fallback to local db on connection error', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    spySettings.dataChange.emit({
      backend: 'cpu',
      backendAuto: true,
      download: true
    });

    spyDb.getSymbols.and.returnValue(Promise.resolve([]));

    const req = httpMock.expectOne('api/symbol/?format=json');
    expect(req.request.method).toBe('GET');
    req.flush('', { status: 400, statusText: '' });
  });

  it('should handle symbol images', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    spySettings.dataChange.emit({
      backend: 'cpu',
      backendAuto: true,
      download: true
    });

    const req = httpMock.expectOne('api/symbol/?format=json');
    expect(req.request.method).toBe('GET');
    req.flush([
      { image: 'abcd' }
    ]);
  });

  it('should get symbols locally', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    spyDb.getSymbols.and.returnValue(Promise.resolve([ { image: 'abcd' } ] as any));

    service.getSymbolsLocal();
  });

  it('should update symbols', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    const symbol = {
      id: 1,
      timestamp: '',
      name: 'test',
      description: 'testdescription',
      latex: '\\test',
      image: ''
    };

    service.updateSymbol(symbol).subscribe(x => {
    });

    const req = httpMock.expectOne(`api/symbol/${symbol.id}/?format=json`);
    expect(req.request.method).toBe('PUT');
    req.flush(symbol);
  });

  it('should update symbol images', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    const symbol = {
      id: 1,
      timestamp: '',
      name: 'test',
      description: 'testdescription',
      latex: '\\test',
      image: ''
    };

    service.updateImage(symbol).subscribe(x => {
    });

    const req = httpMock.expectOne(`api/symbol/1/image/?format=json`);
    expect(req.request.method).toBe('PUT');
    req.flush({});
  });

  it('should create symbols', () => {
    const service: SymbolService = TestBed.get(SymbolService);

    const symbol = {
      timestamp: '',
      name: 'test',
      description: 'testdescription',
      latex: '\\test',
      image: ''
    };

    service.create(symbol).subscribe(x => {
    });

    const req = httpMock.expectOne(`api/symbol/?format=json`);
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  afterEach(() => {
    httpMock.verify();
  });
});
