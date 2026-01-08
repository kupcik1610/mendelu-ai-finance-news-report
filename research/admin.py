from django.contrib import admin
from .models import CompanyResearch, Article


class ArticleInline(admin.TabularInline):
    model = Article
    extra = 0
    readonly_fields = ['url', 'title', 'source', 'ensemble_score', 'sentiment_label']
    fields = ['title', 'source', 'ensemble_score', 'sentiment_label']


@admin.register(CompanyResearch)
class CompanyResearchAdmin(admin.ModelAdmin):
    list_display = ['company_name', 'status', 'overall_sentiment', 'sentiment_label', 'created_at']
    list_filter = ['status', 'sentiment_label', 'created_at']
    search_fields = ['company_name']
    readonly_fields = ['created_at']
    inlines = [ArticleInline]


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'source', 'source_type', 'ensemble_score', 'sentiment_label']
    list_filter = ['sentiment_label', 'source_type']
    search_fields = ['title', 'source']
