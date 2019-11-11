// src/app/auth/token.interceptor.tsimport { Injectable } from '@angular/core';
import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { CookieService } from 'ngx-cookie-service';
import { Observable } from 'rxjs';

@Injectable()
export class Interceptor implements HttpInterceptor {

  constructor(private cookieService: CookieService) {}

  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const csrfToken = this.cookieService.get('csrftoken');

    request = request.clone({
      setHeaders: {
        'X-CSRFToken': csrfToken
      }
    });

    return next.handle(request);
  }
}
