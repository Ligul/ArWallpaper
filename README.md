[![YouTube](https://img.youtube.com/vi/Vdd4WCfk0Yc/0.jpg)](https://www.youtube.com/watch?v=Vdd4WCfk0Yc)
# Сборка
### Windows
- TBD
### Linux
- Склонировать репозиторий

      git clone git@github.com:Ligul/ArWallpaper.git
- Установить зависимости

      sudo pacman -S base-devel opencv hdf5 vtk glfw-x11 freeglut glew qt5-base
- Собрать и установить

      make
      sudo make install
#### или
- Склонировать репозиторий
- Установить docker
- Собрать и установить

      sudo docker build . -t arwallpaper
      sudo docker run -v (pwd):/build arwallpaper
      sudo make install

# Как запустить
### Windows
https://docs.google.com/document/d/10S9xSUz-MUovEwJm-CTkqPraceYm5TXCP17HlU6oSsI/edit
### Linux
Для работы требуется программа devilspie2! После установки ArWallpaper должен появиться в меню приложений
(если всё-таки нет, нужно перезапустить сеанс пользователя).
Проверялось в сеансах KDE Plasma и GNOME(X11)
