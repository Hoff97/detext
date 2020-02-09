import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { NewSymbolComponent } from './new-symbol.component';
import { FormsModule } from '@angular/forms';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('NewSymbolComponent', () => {
  let component: NewSymbolComponent;
  let fixture: ComponentFixture<NewSymbolComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ NewSymbolComponent ],
      imports: [ FormsModule, HttpClientTestingModule ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(NewSymbolComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should be able to reset', () => {
    component.cancel();
  });

  it('should be able to display latex codes', () => {
    component.getLatexSvg();
  });

  it('should be able to create a new class', () => {
    let clsRes;

    component.class.name = 'Test';

    component.created.subscribe(cls => clsRes = cls);

    component.create();

    expect(clsRes).toBeTruthy();
    expect(clsRes.name).toEqual('Test');
  });

  it('should be able to handle image upload', () => {
    component.handleImageInput([new Blob()]);
  });
});
