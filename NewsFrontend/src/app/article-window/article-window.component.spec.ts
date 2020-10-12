import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ArticleWindowComponent } from './article-window.component';

describe('ArticleWindowComponent', () => {
  let component: ArticleWindowComponent;
  let fixture: ComponentFixture<ArticleWindowComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ArticleWindowComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ArticleWindowComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
