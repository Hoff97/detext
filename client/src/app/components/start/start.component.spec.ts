import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { StartComponent } from './start.component';
import { CanvasComponent } from '../canvas/canvas.component';
import { ClassificationComponent } from '../classification/classification.component';
import { FormsModule } from '@angular/forms';
import { NewSymbolComponent } from '../new-symbol/new-symbol.component';
import { ClassSymbolComponent } from '../class-symbol/class-symbol.component';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { SettingsService } from 'src/app/services/settings.service';
import { DbService } from 'src/app/services/db.service';
import { InferenceService } from 'src/app/services/inference.service';
import { TrainImageService } from 'src/app/services/train-image.service';
import { SymbolService } from 'src/app/services/symbol.service';
import { EventEmitter } from '@angular/core';
import { of } from 'rxjs';

describe('StartComponent', () => {
  let component: StartComponent;
  let fixture: ComponentFixture<StartComponent>;

  let inferenceService: jasmine.SpyObj<InferenceService>;
  let trainImageService: jasmine.SpyObj<TrainImageService>;
  let symbolService: jasmine.SpyObj<SymbolService>;

  beforeEach(async(() => {
    inferenceService = jasmine.createSpyObj('InferenceService', ['infer']);
    inferenceService.modelAvailable = new EventEmitter<boolean>();
    inferenceService.model = false;

    trainImageService = jasmine.createSpyObj('TrainImageService', ['create']);
    symbolService = jasmine.createSpyObj('SymbolService', ['getSymbols']);

    TestBed.configureTestingModule({
      declarations: [ StartComponent, CanvasComponent, ClassificationComponent, NewSymbolComponent, ClassSymbolComponent ],
      imports: [ FormsModule, NgbModule, HttpClientTestingModule ],
      providers: [
        { provide: InferenceService, useValue: inferenceService },
        { provide: TrainImageService, useValue: trainImageService },
        { provide: SymbolService, useValue: symbolService }
      ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    symbolService.getSymbols.and.returnValue(of([]));

    fixture = TestBed.createComponent(StartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should react to model loading', () => {
    inferenceService.modelAvailable.emit(true);

    expect(component.modelLoading).toBeFalsy();
  });

  it('should allow predicting the next class', () => {
    let success;

    inferenceService.modelAvailable.emit(true);

    expect(component.modelLoading).toBeFalsy();

    const img = 'img';

    inferenceService.infer.and.returnValue(Promise.resolve([1, 2, 3]) as any);

    component.predictClass(img as any).then(x => {
      success = true;
    });
  });
});
