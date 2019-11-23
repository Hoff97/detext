import { Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges } from '@angular/core';
import { ClassSymbol } from 'src/app/data/types';
import { SymbolService } from 'src/app/services/symbol.service';

interface Prediction {
  prop: number;
  class: ClassSymbol;
}

@Component({
  selector: 'app-classification',
  templateUrl: './classification.component.html',
  styleUrls: ['./classification.component.css']
})
export class ClassificationComponent implements OnInit, OnChanges {

  @Input() public predictions: number[];
  @Input() public classes: ClassSymbol[];
  @Input() public loading: boolean;

  @Output() public correct = new EventEmitter<ClassSymbol>();

  public predSorted: Prediction[];
  public correctSelected = false;

  constructor(private symbolService: SymbolService) { }

  ngOnInit() {
  }

  ngOnChanges(changes: SimpleChanges): void {
    this.correctSelected = false;
    this.predSorted = this.predictions.map((pred, ix) => {
      return {
        prop: pred,
        class: this.classes[ix]
      };
    }).sort((a, b) => a.prop > b.prop ? -1 : 1);
  }

  selectCorrect(cls: ClassSymbol) {
    this.correctSelected = true;
    this.correct.emit(cls);
  }

  created(symbol: ClassSymbol) {
    this.symbolService.create(symbol).subscribe(sym => {
      this.correct.emit(sym);
    });
  }
}
