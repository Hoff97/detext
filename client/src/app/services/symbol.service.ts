import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { ClassSymbol } from '../data/types';
import { DbService } from './db.service';

@Injectable({
  providedIn: 'root'
})
export class SymbolService {

  constructor(private http: HttpClient, private dbService: DbService) { }

  getSymbols(): Observable<ClassSymbol[]> {
    return this.http.get<ClassSymbol[]>('api/symbol/?format=json').pipe(
      map((symbols: ClassSymbol[]) => {
        this.dbService.saveSymbols(symbols);
        return symbols;
      }),
      catchError((err) => {
        return this.dbService.getSymbols();
      })
    );
  }
}
