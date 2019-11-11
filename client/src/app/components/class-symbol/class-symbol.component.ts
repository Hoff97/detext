import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { ClassSymbol } from 'src/app/data/types';

@Component({
  selector: 'app-class-symbol',
  templateUrl: './class-symbol.component.html',
  styleUrls: ['./class-symbol.component.css']
})
export class ClassSymbolComponent implements OnInit {

  @Input() public class: ClassSymbol;
  @Input() public prop: number;
  @Input() public correctEnabled: boolean;

  public correctClass = false;

  @Output() correct = new EventEmitter<ClassSymbol>();

  constructor() { }

  ngOnInit() {
  }

  markCorrect() {
    this.correct.emit(this.class);
    this.correctClass = true;
  }
}
