import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { LoginComponent } from './login.component';
import { FormsModule } from '@angular/forms';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { LoginService } from 'src/app/services/login.service';
import { Observable, of } from 'rxjs';

describe('LoginComponent', () => {
  let component: LoginComponent;
  let fixture: ComponentFixture<LoginComponent>;
  let loginServiceSpy: jasmine.SpyObj<LoginService>;


  beforeEach(async(() => {
    const spy = jasmine.createSpyObj('LoginService', ['login', 'getToken', 'isLoggedIn']);

    TestBed.configureTestingModule({
      declarations: [ LoginComponent ],
      imports: [ FormsModule, HttpClientTestingModule ],
      providers: [
        { provide: LoginService, useValue: spy }
      ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(LoginComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();

    loginServiceSpy = TestBed.get(LoginService);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should handle the login', () => {
    loginServiceSpy.login.and.returnValue(of('testtoken'));
    component.login();
  });
});
