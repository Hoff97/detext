// src/app/auth/token.interceptor.tsimport { Injectable } from '@angular/core';
import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { CookieService } from 'ngx-cookie-service';
import { Observable } from 'rxjs';
import { LoginService } from '../services/login.service';

@Injectable()
export class Interceptor implements HttpInterceptor {

  constructor(private cookieService: CookieService,
              private loginService: LoginService) {}

  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const csrfToken = this.cookieService.get('csrftoken');

    request = request.clone({
      setHeaders: {
        'X-CSRFToken': csrfToken
      }
    });

    const token = this.loginService.getToken();
    if (token !== undefined && token !== null) {
      request = request.clone({
        setHeaders: {
          Authorization: `Token ${token}`
        }
      });
      return next.handle(request);
    }
    return next.handle(request);
  }
}
