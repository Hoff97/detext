import { HttpClientTestingModule } from '@angular/common/http/testing';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule } from '@angular/forms';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { of } from 'rxjs';
import { InferenceService } from 'src/app/services/inference.service';
import { SymbolService } from 'src/app/services/symbol.service';
import { TrainImageService } from 'src/app/services/train-image.service';
import { ClassSymbolComponent } from '../class-symbol/class-symbol.component';
import { ClassificationComponent } from '../classification/classification.component';
import { NewSymbolComponent } from '../new-symbol/new-symbol.component';
import { StartComponent } from './start.component';


@Component({
  selector: 'app-canvas',
  template: '<p>Mock App Canvas Component</p>'
})
class MockCanvasComponent {
  public static imageChange = new EventEmitter<ImageData>();

  @Input() public width = 400;
  @Input() public height = 400;
  @Input() public strokeSize = 3;

  @Output() imageChange = MockCanvasComponent.imageChange;
  @Output() cleared = new EventEmitter<void>();
}

describe('StartComponent', () => {
  let component: StartComponent;
  let fixture: ComponentFixture<StartComponent>;

  let inferenceService: jasmine.SpyObj<InferenceService>;
  let trainImageService: jasmine.SpyObj<TrainImageService>;
  let symbolService: jasmine.SpyObj<SymbolService>;

  beforeEach(async(() => {
    inferenceService = jasmine.createSpyObj('InferenceService', ['infer']);
    inferenceService.modelAvailable = new EventEmitter<boolean>();
    inferenceService.updating = new EventEmitter<boolean>();
    inferenceService.model = false;

    trainImageService = jasmine.createSpyObj('TrainImageService', ['create']);
    symbolService = jasmine.createSpyObj('SymbolService', ['getSymbols']);

    TestBed.configureTestingModule({
      declarations: [ StartComponent, MockCanvasComponent, ClassificationComponent, NewSymbolComponent, ClassSymbolComponent ],
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

  it('should allow predicting the next class', () => {
    inferenceService.modelAvailable.emit(true);

    expect(component.modelLoading).toBeFalsy();

    const img = {
      data: new Uint8ClampedArray(1),
      height: 1,
      width: 1
    };

    inferenceService.infer.and.returnValue(Promise.resolve([1, 2, 3]) as any);

    component.predictClass(img as any).then(x => {
    });

    trainImageService.create.and.returnValue(of({} as any));

    component.correctSelected({} as any);
  });

  it('should allow clearing results', () => {
    component.cleared();

    expect(component.predictions).toEqual([]);
  });

  it('should allow reloading classes', () => {
    symbolService.getSymbols.and.returnValue(of([]));

    component.reloadClasses().then(x => {});
  });

  it('should react to the model being updated', () => {
    inferenceService.updating.emit(false);

    expect(component.modelUpdating).toBeFalsy();
  });
});
