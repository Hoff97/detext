import { TestBed } from '@angular/core/testing';

import { LoginService } from './login.service';

import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';

describe('LoginService', () => {
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [
        HttpClientTestingModule,
      ],
    });

    httpMock = TestBed.get(HttpTestingController);
  });

  it('should be created', () => {
    const service: LoginService = TestBed.get(LoginService);
    expect(service).toBeTruthy();
  });

  it('should perform the login', () => {
    const service: LoginService = TestBed.get(LoginService);

    const token = 'yay';
    let succeeded;

    service.login('test', 'test').subscribe(x => {
      expect(x).toEqual(token);
    });

    service.loginSucceeded.subscribe(x => {
      succeeded = true;
    });

    const req = httpMock.expectOne(`api/api-token-auth/`);
    expect(req.request.method).toBe('POST');
    req.flush({
      token
    });

    expect(succeeded).toBeTruthy();
  });

  it('should not log in on error', () => {
    const service: LoginService = TestBed.get(LoginService);

    const token = 'yay';
    let succeeded;

    service.login('test', 'test').subscribe(x => {
      fail();
    }, err => {
    });

    service.loginSucceeded.subscribe(x => {
      succeeded = true;
    });

    const req = httpMock.expectOne(`api/api-token-auth/`);
    expect(req.request.method).toBe('POST');
    req.flush('', { status: 400, statusText: '' });

    expect(succeeded).toBeFalsy();
  });

  it('should return token after login', () => {
    const service: LoginService = TestBed.get(LoginService);

    const token = 'yay';

    service.login('test', 'test').subscribe(x => {
    });

    const req = httpMock.expectOne(`api/api-token-auth/`);
    expect(req.request.method).toBe('POST');
    req.flush({
      token
    });

    expect(service.getToken()).toEqual(token);
  });

  afterEach(() => {
    httpMock.verify();
  });
});
