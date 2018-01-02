#-------------------------------------------------
#
# Project created by QtCreator 2017-11-19T12:00:30
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OMR
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect

SOURCES += \
        main.cpp \
        OMR.cpp \
    svm.cpp

HEADERS += \
        OMR.h \
    score_class.h \
    svm.h

FORMS += \
        mainwindow.ui

DISTFILES += \
    Train_Data/element1.jpg \
    Train_Data/element10.jpg \
    Train_Data/element11.jpg \
    Train_Data/element12.jpg \
    Train_Data/element13.jpg \
    Train_Data/element14.jpg \
    Train_Data/element15.jpg \
    Train_Data/element16.jpg \
    Train_Data/element17.jpg \
    Train_Data/element18.jpg \
    Train_Data/element19.jpg \
    Train_Data/element2.jpg \
    Train_Data/element20.jpg \
    Train_Data/element21.jpg \
    Train_Data/element22.jpg \
    Train_Data/element23.jpg \
    Train_Data/element24.jpg \
    Train_Data/element25.jpg \
    Train_Data/element26.jpg \
    Train_Data/element27.jpg \
    Train_Data/element28.jpg \
    Train_Data/element3.jpg \
    Train_Data/element4.jpg \
    Train_Data/element5.jpg \
    Train_Data/element6.jpg \
    Train_Data/element7.jpg \
    Train_Data/element8.jpg \
    Train_Data/element9.jpg
