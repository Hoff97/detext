import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { ClassSymbol } from 'src/app/data/types';
import { LoginService } from 'src/app/services/login.service';
import { SymbolService } from 'src/app/services/symbol.service';
import { strToBase64 } from 'src/app/util/data';

@Component({
  selector: 'app-class-symbol',
  templateUrl: './class-symbol.component.html',
  styleUrls: ['./class-symbol.component.css']
})
export class ClassSymbolComponent implements OnInit {

  @Input() public class: ClassSymbol;
  @Input() public prop: number;
  @Input() public correctEnabled: boolean;

  public expanded = false;

  public correctClass = false;

  public loggedIn;

  public descriptEdit = false;
  public latexEdit = false;

  private reader: FileReader;

  @Output() correct = new EventEmitter<ClassSymbol>();

  constructor(private loginService: LoginService,
              private symbolService: SymbolService) {
    this.loggedIn = this.loginService.isLoggedIn();
    this.reader = new FileReader();
  }

  ngOnInit() {
  }

  markCorrect() {
    this.correct.emit(this.class);
    this.correctClass = true;
  }

  toggleExpand() {
    this.expanded = !this.expanded;
    if (!this.expanded) {
      this.descriptEdit = false;
    }
  }

  editDescription() {
    this.descriptEdit = !this.descriptEdit;

    if (!this.descriptEdit) {
      this.symbolService.updateSymbol(this.class).subscribe(response => {
        console.log(response);
      });
    }
  }

  editLatex() {
    this.latexEdit = !this.latexEdit;

    if (!this.latexEdit) {
      this.symbolService.updateSymbol(this.class).subscribe(response => {
        console.log(response);
      });
    }
  }

  handleImageInput(files) {
    this.reader.onload = () => {
      const text = this.reader.result;

      this.class.image = strToBase64(text);
      this.class.imgDatUri = text as any;
      this.symbolService.updateImage(this.class).subscribe(response => {
        console.log(response);
      });
    };

    this.reader.readAsDataURL(files[0]);
  }
}
