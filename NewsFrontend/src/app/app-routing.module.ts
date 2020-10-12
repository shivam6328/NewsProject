import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { SearchPageComponent } from '../app/search-page/search-page.component';
import { SearchAreaComponent } from '../app/search-area/search-area.component';
import { ArticleDisplayComponent } from '../app/article-display/article-display.component';

const routes: Routes = [
  { path: 'analyze', component: SearchPageComponent},
  { path: 'home', component: ArticleDisplayComponent},
  { path: '*', component: SearchPageComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
