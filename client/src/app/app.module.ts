import { HttpClientModule } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BrowserModule } from '@angular/platform-browser';
import { ServiceWorkerModule } from '@angular/service-worker';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { environment } from '../environments/environment';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { CanvasComponent } from './components/canvas/canvas.component';
import { ClassificationComponent } from './components/classification/classification.component';
import { NotFoundComponent } from './components/not-found/not-found.component';
import { SettingsComponent } from './components/settings/settings.component';
import { StartComponent } from './components/start/start.component';
import { ClassSymbolComponent } from './components/class-symbol/class-symbol.component';


@NgModule({
  declarations: [
    AppComponent,
    StartComponent,
    SettingsComponent,
    NotFoundComponent,
    CanvasComponent,
    ClassificationComponent,
    ClassSymbolComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ServiceWorkerModule.register('ngsw-worker.js', { enabled: environment.production }),
    NgbModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
