import { TestBed } from '@angular/core/testing';

import { InferenceService } from './inference.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('InferenceService', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [
      HttpClientTestingModule,
    ],
  }));

  it('should be created', () => {
    const service: InferenceService = TestBed.get(InferenceService);
    expect(service).toBeTruthy();
  });
});
