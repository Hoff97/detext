import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map } from 'rxjs/operators';
import { environment } from 'src/environments/environment';

type token = string;

interface LoginRequest {
  token: token;
}

@Injectable({
  providedIn: 'root'
})
export class LoginService {
  public static storageItem = 'auth-token';

  private loggedIn = false;
  private token?: token;

  private urlPrefix = environment.urlPrefix;

  constructor(private httpClient: HttpClient) {
    const localStorageToken = localStorage.getItem(LoginService.storageItem);
    if (localStorageToken) {
      this.setLoginToken(localStorageToken);
    }
  }

  public login(username: string, password: string) {
    const observable = this.httpClient.post<LoginRequest>(this.urlPrefix + 'api/api-token-auth/', {
      username,
      password
    });

    return observable.pipe(map(response => {
      this.setLoginToken(response.token);
      return this.token;
    }));
  }

  private setLoginToken(tk: string) {
    this.loggedIn = true;
    this.token = tk;
    localStorage.setItem(LoginService.storageItem, tk);
  }

  public getToken() {
    return this.token;
  }

  public isLoggedIn() {
    return this.loggedIn;
  }
}
