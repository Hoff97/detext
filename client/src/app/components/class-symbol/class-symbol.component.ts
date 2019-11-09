import { Component, Input, OnInit } from '@angular/core';
import { ClassSymbol } from 'src/app/data/types';

@Component({
  selector: 'app-class-symbol',
  templateUrl: './class-symbol.component.html',
  styleUrls: ['./class-symbol.component.css']
})
export class ClassSymbolComponent implements OnInit {

  @Input() public class: ClassSymbol;
  @Input() public prop: number;

  constructor() { }

  ngOnInit() {
  }

}
