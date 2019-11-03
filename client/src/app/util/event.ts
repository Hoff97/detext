import { merge, Observable } from 'rxjs';
import { fromEvent, FromEventTarget } from 'rxjs/internal/observable/fromEvent';

export function fromEvents<T>(target: FromEventTarget<T>, ...eventNames: string[]): Observable<T> {
  return merge(
    ...eventNames.map(name => fromEvent(target, name))
  );
}
