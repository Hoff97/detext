import { HttpClient } from '@angular/common/http';
import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import * as moment from 'moment';
import { ClassSymbol } from 'src/app/data/types';
import { LoginService } from 'src/app/services/login.service';
import { strToBase64 } from 'src/app/util/data';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-new-symbol',
  templateUrl: './new-symbol.component.html',
  styleUrls: ['./new-symbol.component.css']
})
export class NewSymbolComponent implements OnInit {

  private urlPrefix = environment.urlPrefix;

  @Output() created = new EventEmitter<ClassSymbol>();

  creating = false;

  class: ClassSymbol = {
    name: '',
    latex: '',
    description: '',
    image: '',
    imgDatUri: '',
    timestamp: moment().format()
  };

  displayCodeImg = false;

  reader = new FileReader();

  public loggedIn;

  constructor(private loginService: LoginService,
              private sanitizer: DomSanitizer,
              private http: HttpClient) {
    this.loggedIn = this.loginService.isLoggedIn();
  }

  ngOnInit() {
  }

  handleImageInput(files) {
    this.reader.onload = () => {
      const text = this.reader.result;

      this.class.image = strToBase64(text);
      this.class.imgDatUri = this.sanitizer.bypassSecurityTrustResourceUrl(text as any) as any;
    };

    this.reader.readAsDataURL(files[0]);
  }

  create() {
    this.created.emit(this.class);
  }

  getLatexSvg() {
    this.displayCodeImg = true;
  }
}
