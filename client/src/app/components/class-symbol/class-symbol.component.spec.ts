import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassSymbolComponent } from './class-symbol.component';

describe('ClassSymbolComponent', () => {
  let component: ClassSymbolComponent;
  let fixture: ComponentFixture<ClassSymbolComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ClassSymbolComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ClassSymbolComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
