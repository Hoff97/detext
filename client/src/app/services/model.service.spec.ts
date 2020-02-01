import { TestBed } from '@angular/core/testing';

import { ModelService } from './model.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('ModelService', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [
      HttpClientTestingModule,
    ],
  }));

  it('should be created', () => {
    const service: ModelService = TestBed.get(ModelService);
    expect(service).toBeTruthy();
  });
});
