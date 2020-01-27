import { Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges } from '@angular/core';
import { ClassSymbol } from 'src/app/data/types';
import { SymbolService } from 'src/app/services/symbol.service';

interface Prediction {
  prop?: number;
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

  @Output() public reloadClasses = new EventEmitter<void>();
  @Output() public correct = new EventEmitter<ClassSymbol>();

  public predSorted: Prediction[];
  public unpredicted: Prediction[];
  public correctSelected = false;

  public tab = 'predicted';

  public shown: Prediction[];
  public page = 1;
  public currentPage: Prediction[];

  public pageSize = 5;

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
    this.unpredicted = this.classes.slice(this.predictions.length).map(cls => {
      return {
        class: cls
      };
    });
    this.setTab(this.tab);
    this.setPage(this.page);
  }

  selectCorrect(cls: ClassSymbol) {
    this.correctSelected = true;
    this.correct.emit(cls);
  }

  created(symbol: ClassSymbol) {
    this.symbolService.create(symbol).subscribe(sym => {
      this.correct.emit(sym);
      this.reloadClasses.emit();
    });
  }

  setTab(tab: string) {
    const changed = this.tab !== tab;

    this.tab = tab;

    if (tab === 'predicted') {
      this.shown = this.predSorted;
    } else {
      this.shown = this.unpredicted;
    }

    if (changed) {
      this.setPage(1);
    }
  }

  setPage(page: number) {
    this.page = page;
    this.currentPage = this.shown.slice((page - 1) * this.pageSize, page * this.pageSize);
  }
}
