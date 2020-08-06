import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassificationComponent } from './classification.component';
import { NewSymbolComponent } from '../new-symbol/new-symbol.component';
import { ClassSymbolComponent } from '../class-symbol/class-symbol.component';
import { FormsModule } from '@angular/forms';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { SymbolService } from 'src/app/services/symbol.service';
import { ClassSymbol } from 'src/app/data/types';
import { of } from 'rxjs';

function getClass(name: string): ClassSymbol {
  return {
    name,
    description: 'test',
    latex: 'test',
    image: 'abcd'
  };
}

describe('ClassificationComponent', () => {
  let component: ClassificationComponent;
  let fixture: ComponentFixture<ClassificationComponent>;
  let symbolServiceSpy: jasmine.SpyObj<SymbolService>;

  beforeEach(async(() => {
    symbolServiceSpy = jasmine.createSpyObj('SymbolService', ['updateSymbol', 'updateImage', 'create']);

    TestBed.configureTestingModule({
      declarations: [ ClassificationComponent, NewSymbolComponent, ClassSymbolComponent ],
      imports: [ FormsModule, NgbModule, HttpClientTestingModule ],
      providers: [
        { provide: SymbolService, useValue: symbolServiceSpy }
      ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ClassificationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should handle changes', () => {
    component.predictions = [1, 2, -1];
    component.uncertainties = [0.001, 0.0001, 0.0001];
    component.classes = [getClass('test1'), getClass('test2'),
                         getClass('test3'), getClass('test4')];

    fixture.detectChanges();

    component.ngOnChanges({} as any);

    component.uncertainties = [0.001, 0.006, 0.0001];
    fixture.detectChanges();
    component.ngOnChanges({} as any);

    component.uncertainties = [0.001, 0.03, 0.0001];
    fixture.detectChanges();
    component.ngOnChanges({} as any);

    component.uncertainties = [0.001, 0.075, 0.0001];
    fixture.detectChanges();
    component.ngOnChanges({} as any);

    component.uncertainties = [0.001, 0.2, 0.0001];
    fixture.detectChanges();
    component.ngOnChanges({} as any);
  });

  it('should change tabs', () => {
    component.predictions = [1, 2, -1];
    component.classes = [getClass('test1'), getClass('test2'),
                         getClass('test3'), getClass('test4')];

    fixture.detectChanges();

    component.ngOnChanges({} as any);

    component.setTab('unpredicted');
  });

  it('should handle selecting the correct symbol', () => {
    let success;

    component.correct.subscribe(x => {
      success = true;
    });

    component.selectCorrect(getClass('test1'));

    expect(component.correctSelected).toBeTruthy();
    expect(success).toBeTruthy();
  });

  it('should handle creating a new symbol', () => {
    let correct;
    let reloadClasses;

    component.correct.subscribe(x => {
      correct = true;
    });
    component.reloadClasses.subscribe(x => {
      reloadClasses = true;
    });

    const symbol = getClass('test1');

    symbolServiceSpy.create.and.returnValue(of(symbol));

    component.created(symbol);

    expect(component.correctSelected).toBeTruthy();
    expect(correct && reloadClasses).toBeTruthy();
  });
});
