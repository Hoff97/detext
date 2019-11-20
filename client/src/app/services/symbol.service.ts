import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { ClassSymbol } from '../data/types';
import { DbService } from './db.service';
import { Settings, SettingsService } from './settings.service';

@Injectable({
  providedIn: 'root'
})
export class SymbolService {

  private urlPrefix = environment.urlPrefix;

  private downloaded = false;

  constructor(private http: HttpClient,
              private dbService: DbService,
              private settingsService: SettingsService) {
    this.settingsService.dataChange.subscribe((data: Settings) => {
      if (data.download && !this.downloaded) {
        this.getSymbols().subscribe(x => {
          this.downloaded = data.download;
        });
      }
    });
  }

  getSymbols(): Observable<ClassSymbol[]> {
    return this.http.get<ClassSymbol[]>(this.urlPrefix + 'api/symbol/?format=json').pipe(
      map((symbols: ClassSymbol[]) => {
        this.dbService.saveSymbols(symbols);
        return symbols;
      }),
      catchError((err) => {
        return this.dbService.getSymbols();
      })
    );
  }

  getSymbolsLocal(): Promise<ClassSymbol[]> {
    return this.dbService.getSymbols();
  }

  updateSymbol(symbol: ClassSymbol) {
    return this.http.put<ClassSymbol[]>(`${this.urlPrefix}api/symbol/${symbol.id}/?format=json`, {
      timestamp: symbol.timestamp,
      name: symbol.name,
      description: symbol.description,
      latex: symbol.latex
    });
  }
}
