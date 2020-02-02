import { async, ComponentFixture, TestBed, tick, fakeAsync } from '@angular/core/testing';

import { CanvasComponent } from './canvas.component';
import { FormsModule } from '@angular/forms';
import { DebugElement, ElementRef } from '@angular/core';
import { By } from '@angular/platform-browser';

describe('CanvasComponent', () => {
  let component: CanvasComponent;
  let fixture: ComponentFixture<CanvasComponent>;

  let canvas: HTMLCanvasElement;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ CanvasComponent ],
      imports: [ FormsModule ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(CanvasComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();

    const hostElement = fixture.nativeElement;
    canvas = hostElement.querySelector('canvas');
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should give the canvas contents', () => {
    let image;

    component.imageChange.subscribe(data => image = data);

    component.emitContent();

    expect(image).toBeTruthy();
  });

  it('should output an image on canvas change', () => {
    let image;

    component.imageChange.subscribe(data => {
      image = data;
    });

    canvas.dispatchEvent(new Event('mouseup'));
    fixture.detectChanges();

    expect(image).toBeTruthy();
  });

  it('should emit cleared when canvas is reset', () => {
    let cleared = false;

    component.cleared.subscribe(data => {
      cleared = true;
    });

    component.clear();
    fixture.detectChanges();

    expect(cleared).toBeTruthy();
  });

  it('should be able to draw on canvas', () => {
    const pos1 = { clientX: 5, clientY: 5};
    const pos2 = { clientX: 10, clientY: 10};

    const ev1 = new MouseEvent('mousemove', pos1);
    const ev2 = new MouseEvent('mousemove', pos2);

    component.handleLineMove([ev1, ev2]);

    component.handleLineMove([{ touches: [ pos1 ] } as any, { touches: [ pos2 as any ] } as any]);
  });
});
