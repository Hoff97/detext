import { Component, OnInit } from '@angular/core';
import { LoginService } from 'src/app/services/login.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  public username = '';
  public password = '';

  returnUrl: string;

  public loggedIn;

  constructor(private loginService: LoginService) { }

  ngOnInit() {
    this.loggedIn = this.loginService.isLoggedIn();
  }

  public login() {
    this.loginService.login(this.username, this.password).subscribe(res => {
      if (res !== undefined) {
        this.loggedIn = this.loginService.isLoggedIn();
      }
    });
  }

}
