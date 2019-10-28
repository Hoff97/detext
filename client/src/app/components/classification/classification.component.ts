import { Component, Input, OnChanges, OnInit, SimpleChanges } from '@angular/core';

interface Prediction {
  prop: number;
  name: string;
}

@Component({
  selector: 'app-classification',
  templateUrl: './classification.component.html',
  styleUrls: ['./classification.component.css']
})
export class ClassificationComponent implements OnInit, OnChanges {

  @Input() public predictions: number[];
  @Input() public classes: string[];
  @Input() public loading: boolean;

  public predSorted: Prediction[];

  constructor() { }

  ngOnInit() {
  }

  ngOnChanges(changes: SimpleChanges): void {
    this.predSorted = this.predictions.map((pred, ix) => {
      return {
        prop: pred,
        name: this.classes[ix]
      };
    }).sort((a, b) => a.prop > b.prop ? -1 : 1);
  }

}
