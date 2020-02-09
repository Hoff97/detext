import { AfterViewInit, Component, ElementRef, EventEmitter, Input, Output, ViewChild } from '@angular/core';
import { pairwise, switchMap, takeUntil } from 'rxjs/operators';
import { fromEvents } from 'src/app/util/event';


type Point = {
  x: number,
  y: number
};

@Component({
  selector: 'app-canvas',
  templateUrl: './canvas.component.html',
  styleUrls: ['./canvas.component.css']
})
export class CanvasComponent implements AfterViewInit {
  // a reference to the canvas element from our template
  @ViewChild('canvas', {static: false}) public canvas: ElementRef<HTMLCanvasElement>;

  // setting a width and height for the canvas
  @Input() public width = 400;
  @Input() public height = 400;
  @Input() public strokeSize = 3;

  @Output() imageChange = new EventEmitter<ImageData>();
  @Output() cleared = new EventEmitter<void>();

  public autoemit = true;

  private cx: CanvasRenderingContext2D;

  public ngAfterViewInit() {
    const canvasEl = this.canvas.nativeElement;
    this.cx = canvasEl.getContext('2d');

    canvasEl.width = this.width;
    canvasEl.height = this.height;

    this.captureEvents(canvasEl);

    this.clear();
  }

  public clear() {
    this.cx.fillStyle = '#FFF';
    this.cx.beginPath();
    this.cx.rect(0, 0, 500, 500);
    this.cx.fill();

    this.cleared.emit();
  }

  private getImageContent(): ImageData {
    return this.cx.getImageData(0, 0, this.width, this.height);
  }

  private captureEvents(canvasEl: HTMLCanvasElement) {
    const mousedown = fromEvents(canvasEl, 'mousedown', 'touchstart');
    const mousemove = fromEvents(canvasEl, 'mousemove', 'touchmove');
    const mouseup = fromEvents(canvasEl, 'mouseup', 'touchend');
    const mouseleave = fromEvents(canvasEl, 'mouseleave', 'touchcancel');

    mousedown
      .pipe(
        switchMap((e) => {
          return mousemove
            .pipe(
              takeUntil(mouseup),
              takeUntil(mouseleave),
              pairwise()
            );
        })
      )
      .subscribe(res => this.handleLineMove(res as any));

    mouseup.subscribe((ev: MouseEvent) => {
      if (this.autoemit) {
        this.emitContent();
      }
    });
  }

  public handleLineMove(res: [MouseEvent | TouchEvent, MouseEvent | TouchEvent]) {
    const rect = this.canvas.nativeElement.getBoundingClientRect();

    let [startX, startY, endX, endY] = [0, 0, 0, 0];

    if (res[0] instanceof MouseEvent) {
      startX = res[0].clientX;
      startY = res[0].clientY;
    } else {
      startX = res[0].touches[0].clientX;
      startY = res[0].touches[0].clientY;
    }
    if (res[1] instanceof MouseEvent) {
      endX = res[1].clientX;
      endY = res[1].clientY;
    } else {
      endX = res[1].touches[0].clientX;
      endY = res[1].touches[0].clientY;
    }

    const prevPos = {
      x: startX - rect.left,
      y: startY - rect.top
    };

    const currentPos = {
      x: endX - rect.left,
      y: endY - rect.top
    };

    this.drawOnCanvas(prevPos, currentPos);
  }

  public emitContent() {
    this.imageChange.emit(this.getImageContent());
  }

  private drawOnCanvas(prevPos: Point, currentPos: Point) {
    if (!this.cx) { return; }

    // set some default properties about the line
    this.cx.lineWidth = this.strokeSize;
    this.cx.lineCap = 'round';
    this.cx.strokeStyle = '#000';

    this.cx.beginPath();

    if (prevPos) {
      this.cx.moveTo(prevPos.x, prevPos.y);
      this.cx.lineTo(currentPos.x, currentPos.y);
      this.cx.stroke();
    }
  }
}
