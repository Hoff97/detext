import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassificationComponent } from './classification.component';
import { NewSymbolComponent } from '../new-symbol/new-symbol.component';
import { ClassSymbolComponent } from '../class-symbol/class-symbol.component';
import { FormsModule } from '@angular/forms';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('ClassificationComponent', () => {
  let component: ClassificationComponent;
  let fixture: ComponentFixture<ClassificationComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ClassificationComponent, NewSymbolComponent, ClassSymbolComponent ],
      imports: [ FormsModule, NgbModule, HttpClientTestingModule ]
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
});
