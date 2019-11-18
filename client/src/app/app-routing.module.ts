import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './components/login/login.component';
import { NotFoundComponent } from './components/not-found/not-found.component';
import { SettingsComponent } from './components/settings/settings.component';
import { StartComponent } from './components/start/start.component';


const routes: Routes = [
  {
    path: '',
    component: StartComponent,
    pathMatch: 'full'
  },
  {
    path: 'settings',
    component: SettingsComponent
  },
  {
    path: 'login',
    component: LoginComponent
  },
  { path: '**', component: NotFoundComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
