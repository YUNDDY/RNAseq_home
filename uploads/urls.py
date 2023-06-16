from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from uploads.core import views


urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    #url(r'^uploads/pca/$', views.PCA_plot, name='PCA'),
    url(r'^uploads/pca/$', views.PCA_plot, name='PCA'),
    url(r'^uploads/venn/$', views.Venn, name='venn'),
    url(r'^uploads/form/$', views.model_form_upload, name='model_form_upload'),
    url(r'^uploads/filter/$', views.data_filter, name='data_filter'),
    url(r'^uploads/download/', views.download_filtered, name='filter_download'),

    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
