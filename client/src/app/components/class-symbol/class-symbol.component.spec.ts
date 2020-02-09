import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassSymbolComponent } from './class-symbol.component';
import { FormsModule } from '@angular/forms';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { LoginService } from 'src/app/services/login.service';
import { SymbolService } from 'src/app/services/symbol.service';
import { of } from 'rxjs';

describe('ClassSymbolComponent', () => {
  let component: ClassSymbolComponent;
  let fixture: ComponentFixture<ClassSymbolComponent>;
  let loginServiceSpy: jasmine.SpyObj<LoginService>;
  let symbolServiceSpy: jasmine.SpyObj<SymbolService>;

  beforeEach(async(() => {
    loginServiceSpy = jasmine.createSpyObj('LoginService', ['login', 'getToken', 'isLoggedIn']);
    symbolServiceSpy = jasmine.createSpyObj('SymbolService', ['updateSymbol', 'updateImage']);

    TestBed.configureTestingModule({
      declarations: [ ClassSymbolComponent ],
      imports: [ FormsModule, HttpClientTestingModule ],
      providers: [
        { provide: LoginService, useValue: loginServiceSpy },
        { provide: SymbolService, useValue: symbolServiceSpy }
      ]
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

  it('should handle marking correct', () => {
    let success;

    component.correct.subscribe(x => {
      success = true;
    });
    component.markCorrect();

    expect(success).toBeTruthy();
  });

  it('should be expandable', () => {
    component.toggleExpand();

    expect(component.expanded).toBeTruthy();

    component.toggleExpand();

    expect(component.expanded).toBeFalsy();
  });

  it('should allow description editing', () => {
    component.editDescription();

    expect(component.descriptEdit).toBeTruthy();

    symbolServiceSpy.updateSymbol.and.returnValue(of({}) as any);

    component.editDescription();

    expect(component.descriptEdit).toBeFalsy();
  });

  it('should allow latex code editing', () => {
    component.editLatex();

    expect(component.latexEdit).toBeTruthy();

    symbolServiceSpy.updateSymbol.and.returnValue(of({}) as any);

    component.editLatex();

    expect(component.latexEdit).toBeFalsy();
  });

  it('should allow name editing', () => {
    component.editName();

    expect(component.nameEdit).toBeTruthy();

    symbolServiceSpy.updateSymbol.and.returnValue(of({}) as any);

    component.editName();

    expect(component.nameEdit).toBeFalsy();
  });

  it('should handle image input', () => {
    symbolServiceSpy.updateImage.and.returnValue(of({}) as any);

    component.handleImageInput([new Blob()]);
  });
});
