import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { NavBarComponent } from './nav-bar/nav-bar.component';
import { SearchAreaComponent } from './search-area/search-area.component';
import { SearchPageComponent } from './search-page/search-page.component';
import { HomePageComponent } from './home-page/home-page.component';
import { ArticleDisplayComponent } from './article-display/article-display.component';
import { SideNavbarComponent } from './side-navbar/side-navbar.component';
import { ArticleWindowComponent } from './article-window/article-window.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    NavBarComponent,
    SearchAreaComponent,
    SearchPageComponent,
    HomePageComponent,
    ArticleDisplayComponent,
    SideNavbarComponent,
    ArticleWindowComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
