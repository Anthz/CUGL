#-------------------------------------------------
#
# Project created by QtCreator 2015-10-09T17:12:57
#
#-------------------------------------------------

QT       += core gui
CONFIG   += C++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CUGL
TEMPLATE = app

# Source files
SOURCES += main.cpp\
        mainwindow.cpp \
        glwidget.cpp \
        glbuffertab.cpp \
		cubuffertab.cpp \
        controltab.cpp \
        paramtab.cpp \
		kerneltab.cpp \
        shader.cpp \
        object.cpp \
        glgeneraltab.cpp \
        colourtextbox.cpp \
        cuglbuffer.cpp \
        objecttab.cpp \
        cugeneraltab.cpp \
        objectpopup.cpp \
		bufferpopup.cpp \
        outputsettings.cpp \
        glsettings.cpp \
        cusettings.cpp \
		texturetab.cpp \
		texture.cpp \
		texturepopup.cpp \
		utilities.cpp \

HEADERS  += mainwindow.h \
            glwidget.h \
            glbuffertab.h \
			cubuffertab.h \
            controltab.h \
            paramtab.h \
			kerneltab.h \
            shader.h \
            object.h \
            glgeneraltab.h \
            colourtextbox.h \
            cuglbuffer.h \
            objecttab.h \
            cugeneraltab.h \
            objectpopup.h \
			bufferpopup.h \
            outputsettings.h \
            glsettings.h \
            cusettings.h \
			texturetab.h \
			texture.h \
			texturepopup.h \
			utilities.h \
			serializable.h \

FORMS    += mainwindow.ui

OTHER_FILES += Shaders/*.vert \
               Shaders/*.frag \
               Kernels/kernel.cu

CUDA_SOURCES += Kernels/kernel.cu

INCLUDEPATH += $$PWD/
DEPENDPATH += $$PWD/

DESTDIR = debug
OBJECTS_DIR = debug/obj           # directory where .obj files will be saved

#CUDA Settings
CUDA_OBJECTS_DIR = debug/obj      # directory where .obj  of cuda file will be saved

CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5"
SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_35           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS += --use_fast_math # default setting

INCLUDE_FORMATTED = $$join(CUDA_DIR,'"', '"', '/include"')
LIB_FORMATTED = $$join(CUDA_DIR,'"', '"', '/lib/x64"')
NVCC_FORMATTED = $$join(CUDA_DIR,'"', '"', '/bin/nvcc"')

# include paths

INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME #"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/lib/x64"

# Add the necessary libraries
CUDA_LIBS= -lcuda -lcudart
#add quotation for those directories contain space (Windows required)
CUDA_INC +=$$join(INCLUDEPATH,'" -I"','-I"','"')

LIBS += $$CUDA_LIBS

#nvcc config
# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

CONFIG(debug, debug|release) {
    #Debug settings
    # Debug mode
    cuda_d.input    = CUDA_SOURCES
    cuda_d.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$NVCC_FORMATTED -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -G -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
     # Release settings
     cuda.input    = CUDA_SOURCES
     cuda.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
     cuda.commands = $$NVCC_FORMATTED $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    --compile -cudart static -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
     cuda.dependency_type = TYPE_C
     QMAKE_EXTRA_COMPILERS += cuda
}

