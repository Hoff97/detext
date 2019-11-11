import { TestBed } from '@angular/core/testing';

import { TrainImageService } from './train-image.service';

describe('TrainImageService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: TrainImageService = TestBed.get(TrainImageService);
    expect(service).toBeTruthy();
  });
});
