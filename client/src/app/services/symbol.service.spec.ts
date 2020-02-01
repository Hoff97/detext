import { TestBed } from '@angular/core/testing';

import { SymbolService } from './symbol.service';

import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';

describe('SymbolService', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [
      HttpClientTestingModule,
    ],
  }));

  it('should be created', () => {
    const service: SymbolService = TestBed.get(SymbolService);
    expect(service).toBeTruthy();
  });
});
