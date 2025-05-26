from django.contrib import admin
from django.urls import path
from . import views



urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name = "home"),
    path('signup/',views.signup,name = "signup"),
    path('login/',views.login_page,name="login"),
    path('logout/',views.logout_page,name="logout"),
    # path('acc_home/',views.acc_home,name = "acc_home"),
    # path('predict/',views.url_predict,name = "predict"),
    path('contact/',views.contactus,name = "contactus"),
    path('contactus_form/',views.contactus_form,name="contactus_form"),
    path('dataset/',views.dataset,name = "dataset"),
    path('project_home/',views.project_home,name = "project_home"),
    path('admin_login/',views.admin_login,name="admin_login"),
    path('admin_home',views.admin_home,name="admin_home"),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('retrain_trigger/',views.retrain_trigger,name="retrain_trigger"),
    path('login_info/', views.login_info, name='login_info'),

]