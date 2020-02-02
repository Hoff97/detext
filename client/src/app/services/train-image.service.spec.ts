import { TestBed } from '@angular/core/testing';

import { TrainImageService } from './train-image.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('TrainImageService', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [ HttpClientTestingModule ]
  }));

  it('should be created', () => {
    const service: TrainImageService = TestBed.get(TrainImageService);
    expect(service).toBeTruthy();
  });
});
