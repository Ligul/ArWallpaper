FROM archlinux:base-devel

RUN pacman -Syyu --noconfirm
RUN pacman -S --noconfirm opencv hdf5 vtk glfw-x11 freeglut glew qt5-base

RUN mkdir /build
WORKDIR /build
CMD make
