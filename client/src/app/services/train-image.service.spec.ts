import { TestBed } from '@angular/core/testing';

import { TrainImageService } from './train-image.service';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';

describe('TrainImageService', () => {
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ HttpClientTestingModule ]
    });

    httpMock = TestBed.get(HttpTestingController);
  });

  it('should be created', () => {
    const service: TrainImageService = TestBed.get(TrainImageService);
    expect(service).toBeTruthy();
  });

  it('should allow train image creation', () => {
    let success;

    const service: TrainImageService = TestBed.get(TrainImageService);

    service.create({} as any).subscribe(x => {
      success = true;
    });

    const req = httpMock.expectOne('api/image/?format=json');
    expect(req.request.method).toBe('POST');
    req.flush([]);
  });
});
