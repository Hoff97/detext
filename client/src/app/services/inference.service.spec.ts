import { TestBed } from '@angular/core/testing';

import { InferenceService } from './inference.service';

describe('InferenceService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: InferenceService = TestBed.get(InferenceService);
    expect(service).toBeTruthy();
  });
});
