import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { StartComponent } from './start.component';
import { CanvasComponent } from '../canvas/canvas.component';
import { ClassificationComponent } from '../classification/classification.component';
import { FormsModule } from '@angular/forms';
import { NewSymbolComponent } from '../new-symbol/new-symbol.component';
import { ClassSymbolComponent } from '../class-symbol/class-symbol.component';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('StartComponent', () => {
  let component: StartComponent;
  let fixture: ComponentFixture<StartComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ StartComponent, CanvasComponent, ClassificationComponent, NewSymbolComponent, ClassSymbolComponent ],
      imports: [ FormsModule, NgbModule, HttpClientTestingModule ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(StartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
